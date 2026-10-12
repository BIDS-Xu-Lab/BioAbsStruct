"""Sentence-level matching between a source abstract and a predicted structured abstract.

Shared by the verbatim-fidelity check (R2-M1) and the section confusion matrix (R1-2).

Core idea: segment the source and the prediction with the same sentence splitter
(pysbd; naive splitting on "." explodes on medical abbreviations), key each
sentence under a comparison mode, then align prediction sentences to source
sentences by exact key match (multiset, order-aware). Leftover unmatched
sentences are paired by fuzzy similarity to separate "modified" (lexical
substitution) from truly "missing" / "added".

Modes:
  strict     - sentence text with only leading/trailing whitespace stripped
  normalized - NFKC + lowercase + all whitespace runs collapsed to one space
"""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

SECTION_ORDER = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
MODES = ("strict", "normalized")

# Unmatched source/pred sentences with similarity >= this are "modified", not missing+added.
MODIFIED_MIN_RATIO = 0.8
# Skip fuzzy pairing when the unmatched grid is huge (degenerate outputs); counts stay missing/added.
MAX_FUZZY_PAIRS = 2500
# Fuzzy split/merge: one sentence vs a run of up to this many consecutive sentences on the other side.
MAX_SPLIT_RUN = 4

_WS_RE = re.compile(r"\s+")
_GLUED_BOUNDARY_RE = re.compile(r"(?<=[a-z]{2}[.!?])(?=[A-Z][a-z])")
# U+0000-U+001F except \t\n\r: never legitimate in abstract text; produced when the model
# mis-writes a \uXXXX JSON escape (e.g. "ö" -> "\u0000f6", "±" -> "\u0000\u0000...").
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# --- modification typing -----------------------------------------------------------
MOD_TYPES = ["encoding", "numbering_header", "connective", "case_punct_whitespace", "paraphrase"]
# Precedence when one sentence pair has several edit types (most substantive wins).
_MOD_PRECEDENCE = ["paraphrase", "encoding", "connective", "numbering_header", "case_punct_whitespace"]
_NUMBERING_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"\(?\s*(?:\d{1,2}|[ivx]{1,4}|[a-h])\s*[.)]"  # "1." "2)" "(a)" "iii."
    r"|[a-z][a-z0-9 /&,-]{1,50}:"  # "lay description:" "results:"
    r")\s*$"
)
_CONNECTIVE_RE = re.compile(
    r"^\s*(?:furthermore|moreover|in addition|additionally|however|notably|interestingly|"
    r"surprisingly|importantly|remarkably|finally|lastly|thus|therefore|hence|also|in conclusion|"
    r"in summary|to summarize|overall|in contrast|conversely|similarly|likewise|consequently|"
    r"accordingly|nevertheless|nonetheless|specifically|in particular|here|herein|first|firstly|"
    r"second|secondly|third|thirdly|then|next|taken together|collectively|altogether|together|"
    r"in this study|in the present study|as a result|of note|indeed|in fact|in general|"
    r"on the other hand|by contrast|in turn|of these|among these)\s*,?\s*$"
)
_PUNCT_WS_RE = re.compile(r"^[\s\W_]*$")


def _edit_type(src_seg: str, pred_seg: str, at_start: bool) -> str:
    if _CTRL_RE.search(pred_seg):
        return "encoding"
    # A non-ASCII char swapped for a short substitute ("μ" -> "u", "–" -> "-"), not a reworded span.
    ascii_left = "".join(c for c in src_seg if ord(c) <= 127)
    if len(ascii_left) < len(src_seg) and len(ascii_left.strip()) <= 2 and len(pred_seg) <= 4:
        return "encoding"
    if _PUNCT_WS_RE.match(src_seg) and _PUNCT_WS_RE.match(pred_seg):
        return "case_punct_whitespace"
    if at_start and not pred_seg.strip():
        if _NUMBERING_HEADER_RE.match(src_seg):
            return "numbering_header"
        if _CONNECTIVE_RE.match(src_seg):
            return "connective"
    return "paraphrase"


def classify_modification(src: str, pred: str) -> tuple[str, list[str]]:
    """Type the edits between a paired source and prediction sentence.

    Returns (primary_type, all_types). Case is ignored so that dropping a leading
    connective ("Furthermore, we ..." -> "We ...") doesn't also count as an edit.
    """
    s, p = src.lower(), pred.lower()
    sm = SequenceMatcher(None, s, p, autojunk=False)
    types = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            types.add(_edit_type(s[i1:i2], p[j1:j2], at_start=(i1 == 0 and j1 == 0)))
    if not types:
        types.add("case_punct_whitespace")
    ordered = [t for t in _MOD_PRECEDENCE if t in types]
    return ordered[0], ordered


# At a split/merge point the model swaps a joining word for a sentence break (", and we" -> ". We").
_BOUNDARY_JOIN_RE = re.compile(
    r"^[\s\W_]*(?:and|but|while|whereas|which|so|thus|and then|as well as)?[\s\W_]*$"
)
# ...optionally re-inserting a subject pronoun ("... and found that" -> "... . We found that").
_BOUNDARY_BREAK_RE = re.compile(r"^[\s\W_]*(?:we|i|they|it|this|these|he|she)?[\s\W_]*$")


def classify_split_merge(src: str, pred: str) -> tuple[str, list[str]]:
    """classify_modification for a joined split/merge group, treating a joining word
    replaced by (or restored from) sentence punctuation as "connective"."""
    s, p = src.lower(), pred.lower()
    sm = SequenceMatcher(None, s, p, autojunk=False)
    types = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a, b = s[i1:i2], p[j1:j2]
        if (_BOUNDARY_JOIN_RE.match(a) and _BOUNDARY_BREAK_RE.match(b)) or (_BOUNDARY_JOIN_RE.match(b) and _BOUNDARY_BREAK_RE.match(a)):
            types.add("case_punct_whitespace" if _PUNCT_WS_RE.match(a) and _PUNCT_WS_RE.match(b) else "connective")
        else:
            types.add(_edit_type(a, b, at_start=(i1 == 0 and j1 == 0)))
    if not types:
        types.add("case_punct_whitespace")
    ordered = [t for t in _MOD_PRECEDENCE if t in types]
    return ordered[0], ordered


@lru_cache(maxsize=1)
def _segmenter():
    import pysbd

    return pysbd.Segmenter(language="en", clean=False)


# Whitespace minus control chars: Python's str.strip()/\s treat \x0b \x0c \x1c-\x1f as
# whitespace, which would silently trim encoding garbage off sentence edges.
_EDGE_WS_RE = re.compile(r"^[^\S\x0b\x0c\x1c-\x1f]+|[^\S\x0b\x0c\x1c-\x1f]+$")
_LEAD_WS_RE = re.compile(r"[^\S\x0b\x0c\x1c-\x1f]*")


def _strip_keep_ctrl(s: str) -> str:
    return _EDGE_WS_RE.sub("", s)


def split_sentences(text: str) -> list[str]:
    """pysbd segmentation; returns stripped, non-empty sentences in order.

    Control chars crash pysbd's numbered-list parser (int('\\x1f6')), so boundaries are
    found on a same-length copy with each control char masked to a space, and the
    sentences are cut from the original text at those offsets. Encoding garbage thus
    survives into the returned sentences and is classified downstream, not hidden.
    """
    if not text or not _strip_keep_ctrl(text):
        return []
    masked = _CTRL_RE.sub(" ", text)
    starts, cur = [], 0
    for seg in _segmenter().segment(masked):
        body = seg.strip()
        if not body:
            continue
        at = masked.find(body, cur)
        if at < 0:
            raise ValueError(f"pysbd segment not found in text at offset {cur}: {body[:40]!r}")
        starts.append(at)
        cur = at + len(body)
    if not starts:
        return []
    starts[0] = 0  # leading whitespace/control chars belong to the first sentence
    out = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        # pysbd misses boundaries with no space after the period ("schizophrenia.In the ...").
        out.extend(p for p in map(_strip_keep_ctrl, _GLUED_BOUNDARY_RE.split(text[a:b])) if p)
    return out


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Same segmentation as split_sentences, as (start, end) character offsets into text:
    [text[s:e] for s, e in split_sentence_spans(text)] == split_sentences(text)."""
    if not text or not _strip_keep_ctrl(text):
        return []
    masked = _CTRL_RE.sub(" ", text)
    starts, cur = [], 0
    for seg in _segmenter().segment(masked):
        body = seg.strip()
        if not body:
            continue
        at = masked.find(body, cur)
        if at < 0:
            raise ValueError(f"pysbd segment not found in text at offset {cur}: {body[:40]!r}")
        starts.append(at)
        cur = at + len(body)
    if not starts:
        return []
    starts[0] = 0
    out = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        cuts = [a] + [a + m.start() for m in _GLUED_BOUNDARY_RE.finditer(text[a:b])] + [b]
        for s, e in zip(cuts, cuts[1:]):
            body = _strip_keep_ctrl(text[s:e])
            if body:
                s += _LEAD_WS_RE.match(text, s, e).end() - s
                out.append((s, s + len(body)))
    return out


def sentence_key(sentence: str, mode: str) -> str:
    if mode == "strict":
        return sentence.strip()
    if mode == "normalized":
        return _WS_RE.sub(" ", unicodedata.normalize("NFKC", sentence)).strip().lower()
    raise ValueError(f"unknown mode: {mode}")


def parse_prediction(pred_json: str | None) -> dict[str, str] | None:
    """Parse predicted_structured_abstract. Returns None for failures ("{}" / invalid)."""
    if pred_json is None:
        return None
    try:
        obj = json.loads(pred_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    return {k: v for k, v in obj.items() if isinstance(v, str) and v.strip()}


def prediction_text(sections: dict[str, str]) -> str:
    """Concatenate predicted sections in canonical order."""
    return " ".join(sections[k].strip() for k in SECTION_ORDER if k in sections)


def labeled_prediction_sentences(sections: dict[str, str]) -> list[tuple[str, str]]:
    """(sentence, section_label) segmented per section, in canonical order (for R1-2)."""
    out = []
    for k in SECTION_ORDER:
        if k in sections:
            out.extend((s, k) for s in split_sentences(sections[k]))
    return out


@dataclass
class Alignment:
    """Result of aligning prediction sentences onto source sentences.

    pred_to_src[j] = source index matched by prediction sentence j (exact key match),
                     or None if j is a duplicate, modified, or added sentence.
    """

    n_src: int
    n_pred: int
    pred_to_src: list[int | None]
    missing_src: list[int] = field(default_factory=list)  # src idx never produced
    added_pred: list[int] = field(default_factory=list)  # pred idx with no source counterpart
    duplicated_pred: list[int] = field(default_factory=list)  # pred idx repeating an already-used source sentence
    modified: list[tuple[int, int, float]] = field(default_factory=list)  # (src idx, pred idx, ratio)
    # Same text modulo whitespace but split/merged differently: (src idxs, pred idxs).
    resegmented: list[tuple[list[int], list[int]]] = field(default_factory=list)
    # Split/merged differently AND lightly edited (joined text ratio >= MODIFIED_MIN_RATIO):
    # (src idxs, pred idxs, ratio). Only filled when align(fuzzy_resegment=True).
    split_merge: list[tuple[list[int], list[int], float]] = field(default_factory=list)
    n_moved: int = 0  # matched pred sentences outside the longest in-order run

    @property
    def reordered(self) -> bool:
        return self.n_moved > 0

    @property
    def exact(self) -> bool:
        return (
            self.n_src == self.n_pred
            and not self.missing_src
            and not self.added_pred
            and not self.duplicated_pred
            and not self.modified
            and not self.resegmented
            and not self.split_merge
            and self.n_moved == 0
        )


def _lis_length(seq: list[int]) -> int:
    tails: list[int] = []
    for x in seq:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _split_merge(unmatched_src, unmatched_pred, src_keys, pred_keys):
    """Best-first greedy: one leftover sentence vs a run of 2..MAX_SPLIT_RUN consecutive
    leftover sentences on the other side, compared as joined text (control chars ignored)."""
    cands = []
    for one, many, one_keys, many_keys, swap in (
        (unmatched_src, unmatched_pred, src_keys, pred_keys, False),
        (unmatched_pred, unmatched_src, pred_keys, src_keys, True),
    ):
        many_set = set(many)
        for a in one:
            target = _CTRL_RE.sub("", one_keys[a])
            for b in many:
                run = [b]
                while len(run) < MAX_SPLIT_RUN and run[-1] + 1 in many_set:
                    run.append(run[-1] + 1)
                    joined = _CTRL_RE.sub("", " ".join(many_keys[k] for k in run))
                    sm = SequenceMatcher(None, target, joined, autojunk=False)
                    if sm.real_quick_ratio() < MODIFIED_MIN_RATIO or sm.quick_ratio() < MODIFIED_MIN_RATIO:
                        continue
                    r = sm.ratio()
                    # Only if joining the run beats the best 1:1 pairing; otherwise the
                    # 1:1 fuzzy pass (modified) is the better explanation.
                    if r >= MODIFIED_MIN_RATIO and r > max(
                        SequenceMatcher(None, target, _CTRL_RE.sub("", many_keys[k]), autojunk=False).ratio() for k in run
                    ):
                        cands.append((r, swap, a, list(run)))
    taken_s, taken_p, out = set(), set(), []
    for r, swap, a, run in sorted(cands, key=lambda c: -c[0]):
        srcs, preds = (run, [a]) if swap else ([a], run)
        if taken_s.intersection(srcs) or taken_p.intersection(preds):
            continue
        taken_s.update(srcs)
        taken_p.update(preds)
        out.append((srcs, preds, r))
    return sorted(out), taken_s, taken_p


def align(src_sents: list[str], pred_sents: list[str], mode: str, fuzzy_resegment: bool = False) -> Alignment:
    src_keys = [sentence_key(s, mode) for s in src_sents]
    pred_keys = [sentence_key(s, mode) for s in pred_sents]

    # Queue of unused source indices per key, so repeated sentences in the source
    # are consumed in order and genuine repeats aren't flagged as duplicates.
    free: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(src_keys):
        free[k].append(i)
    ever_in_src = set(src_keys)

    pred_to_src: list[int | None] = []
    duplicated, unmatched_pred = [], []
    for j, k in enumerate(pred_keys):
        if free.get(k):
            pred_to_src.append(free[k].pop(0))
        else:
            pred_to_src.append(None)
            (duplicated if k in ever_in_src else unmatched_pred).append(j)

    used = {i for i in pred_to_src if i is not None}
    unmatched_src = [i for i in range(len(src_keys)) if i not in used]

    # Boundary artifacts: one side's sentence == consecutive run on the other side,
    # ignoring whitespace (e.g. source "schizophrenia.In the ..." not split by pysbd).
    resegmented = []
    for one, many, one_keys, many_keys, swap in (
        (unmatched_src, unmatched_pred, src_keys, pred_keys, False),
        (unmatched_pred, unmatched_src, pred_keys, src_keys, True),
    ):
        many_set = set(many)
        for a in list(one):
            target = _WS_RE.sub("", one_keys[a])
            for b in many:
                run, acc = [], ""
                k = b
                while k in many_set and target.startswith(acc + _WS_RE.sub("", many_keys[k])):
                    acc += _WS_RE.sub("", many_keys[k])
                    run.append(k)
                    k += 1
                    if acc == target:
                        break
                if len(run) >= 2 and acc == target:
                    one.remove(a)
                    for k in run:
                        many.remove(k)
                        many_set.discard(k)
                    resegmented.append((run, [a]) if swap else ([a], run))
                    break

    # Before 1:1 pairing, so a split sentence's larger half isn't paired alone (other half -> added).
    # Leftovers that are one sentence split into several (or merged) with light edits,
    # e.g. "..., and we examined X." -> "... ." + "We examined X." Otherwise missing + added.
    split_merge = []
    if fuzzy_resegment and unmatched_src and unmatched_pred and len(unmatched_src) * len(unmatched_pred) <= MAX_FUZZY_PAIRS:
        split_merge, taken_s, taken_p = _split_merge(unmatched_src, unmatched_pred, src_keys, pred_keys)
        unmatched_src = [i for i in unmatched_src if i not in taken_s]
        unmatched_pred = [j for j in unmatched_pred if j not in taken_p]

    # Pair leftovers by similarity: best-first greedy.
    modified = []
    if unmatched_src and unmatched_pred and len(unmatched_src) * len(unmatched_pred) <= MAX_FUZZY_PAIRS:
        cands = []
        for i in unmatched_src:
            for j in unmatched_pred:
                # Control chars are always generation garbage; don't let long \u0000 runs block pairing.
                sm = SequenceMatcher(None, _CTRL_RE.sub("", src_keys[i]), _CTRL_RE.sub("", pred_keys[j]), autojunk=False)
                if sm.real_quick_ratio() < MODIFIED_MIN_RATIO or sm.quick_ratio() < MODIFIED_MIN_RATIO:
                    continue
                r = sm.ratio()
                if r >= MODIFIED_MIN_RATIO:
                    cands.append((r, i, j))
        taken_i, taken_j = set(), set()
        for r, i, j in sorted(cands, reverse=True):
            if i in taken_i or j in taken_j:
                continue
            taken_i.add(i)
            taken_j.add(j)
            modified.append((i, j, r))
        unmatched_src = [i for i in unmatched_src if i not in taken_i]
        unmatched_pred = [j for j in unmatched_pred if j not in taken_j]

    matched_seq = [i for i in pred_to_src if i is not None]
    n_moved = len(matched_seq) - _lis_length(matched_seq)

    return Alignment(
        n_src=len(src_keys),
        n_pred=len(pred_keys),
        pred_to_src=pred_to_src,
        missing_src=unmatched_src,
        added_pred=unmatched_pred,
        duplicated_pred=duplicated,
        modified=sorted(modified),
        resegmented=resegmented,
        split_merge=split_merge,
        n_moved=n_moved,
    )


def label_source_sentences(
    src_sents: list[str], sections: dict[str, str], mode: str = "normalized"
) -> list[str | None]:
    """Assign each source sentence the predicted section label it landed in (for R1-2).

    Segments each predicted section separately so labels are exact, then uses the
    same alignment. Modified sentences inherit their pair's label; missing -> None.
    """
    labeled = labeled_prediction_sentences(sections)
    a = align(src_sents, [s for s, _ in labeled], mode)
    out: list[str | None] = [None] * len(src_sents)
    for j, i in enumerate(a.pred_to_src):
        if i is not None:
            out[i] = labeled[j][1]
    for i, j, _ in a.modified:
        out[i] = labeled[j][1]
    return out


# --- control-character repair ------------------------------------------------------
_HEX_ESCAPE_RE = re.compile(r"([\x00-\x08\x0b\x0c\x0e-\x1f])([0-9a-fA-F]{2})")
GAP_MIN_RATIO = 0.5


# An added sentence whose text is (almost) all found inside one source sentence.
REPEAT_MIN_COVER = 0.9


def is_source_repeat(sentence: str, src_sents: list[str]) -> bool:
    """True if an added prediction sentence repeats a fragment of some source sentence
    (e.g. a clause of an already-produced sentence emitted again), rather than new text."""
    t = _WS_RE.sub(" ", _CTRL_RE.sub("", sentence)).strip().lower().rstrip(".")
    if not t:
        return True
    for s in src_sents:
        k = _WS_RE.sub(" ", _CTRL_RE.sub("", s)).lower()
        if t in k:
            return True
        m = SequenceMatcher(None, t, k, autojunk=False)
        if sum(b.size for b in m.get_matching_blocks()) >= REPEAT_MIN_COVER * len(t):
            return True
    return False


def has_control_chars(text: str) -> bool:
    return bool(_CTRL_RE.search(text))


def strip_ctrl(s: str) -> str:
    """Delete the model's garbled escapes ('\\x00' + 'f6') and any other control chars. No guessing."""
    return _CTRL_RE.sub("", _HEX_ESCAPE_RE.sub("", s))


def _trim_to_pred_start(src_text: str, pred_text: str, max_prefix: int = 60) -> str:
    """Drop a short source prefix the model deliberately removed (header "Results ", "1. ")."""
    probe = _CTRL_RE.sub("", pred_text).strip()[:15]
    if len(probe) >= 8:
        at = src_text.find(probe)
        if 0 < at <= max_prefix:
            return src_text[at:]
    return src_text


def repair_control_chars(source: str, sections: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    """Replace prediction sentences containing control-char garbage with their source sentence.

    1. pair:  sentence aligned (fuzzy) to a source sentence -> that source sentence
    2. gap:   run of garbage sentences between two exact-match anchors -> the unused
              source sentences between the anchors (covers '±' runs that ate following text)
    3. strip: no source counterpart -> delete the garbled escapes (no guessed characters); counted
              in stats['stripped_unmatched'] so the record can be flagged
    Sentences without control chars are left untouched.
    """
    stats = {"ctrl_sentences": 0, "fixed_pair": 0, "fixed_gap": 0, "stripped_unmatched": 0}
    if not any(has_control_chars(v) for v in sections.values()):
        return sections, stats

    src = split_sentences(source)
    labeled = labeled_prediction_sentences(sections)
    pred = [s for s, _ in labeled]
    a = align(src, pred, "normalized")

    ctrl_js = [j for j, s in enumerate(pred) if has_control_chars(s)]
    stats["ctrl_sentences"] = len(ctrl_js)
    pair_of = {j: i for i, j, _ in a.modified}
    src_of = {j: i for j, i in enumerate(a.pred_to_src) if i is not None} | pair_of
    used_src = set(src_of.values())

    # j -> (last j of span, replacement text); span covers pred[j..last] (same section).
    edits: dict[int, tuple[int, str]] = {}
    for j in ctrl_js:
        if j in pair_of:
            edits[j] = (j, _trim_to_pred_start(src[pair_of[j]], pred[j]))
            stats["fixed_pair"] += 1

    remaining = [j for j in ctrl_js if j not in edits]
    runs: list[list[int]] = []
    for j in remaining:
        if runs and runs[-1][-1] == j - 1 and labeled[j][1] == labeled[j - 1][1]:
            runs[-1].append(j)
        else:
            runs.append([j])
    for run in runs:
        lo = max((src_of[j] for j in range(run[0]) if j in src_of), default=-1)
        hi = min((src_of[j] for j in range(run[-1] + 1, len(pred)) if j in src_of), default=len(src))
        cands = [i for i in range(lo + 1, hi) if i not in used_src]
        if lo < hi and cands:
            garbled = _CTRL_RE.sub("", " ".join(pred[j] for j in run)).lower()
            target = " ".join(src[i] for i in cands)
            if SequenceMatcher(None, garbled, target.lower(), autojunk=False).ratio() >= GAP_MIN_RATIO:
                edits[run[0]] = (run[-1], _trim_to_pred_start(target, pred[run[0]]))
                used_src.update(cands)
                stats["fixed_gap"] += len(run)
                continue
        for j in run:
            stats["stripped_unmatched"] += 1
            edits[j] = (j, strip_ctrl(pred[j]))

    # Rewrite each section in place so untouched text keeps its original spacing.
    out = dict(sections)
    j0 = 0
    for k in SECTION_ORDER:
        if k not in sections:
            continue
        text = sections[k].strip()
        js = [j for j in range(j0, len(labeled)) if labeled[j][1] == k]
        j0 = js[-1] + 1 if js else j0
        pos, spans, cur = {}, [], 0
        for j in js:
            at = text.find(pred[j], cur)
            if at < 0:
                at = cur
            pos[j] = (at, at + len(pred[j]))
            cur = pos[j][1]
        pieces, cur = [], 0
        for j in js:
            if j in edits:
                last, repl = edits[j]
                start, end = pos[j][0], pos[min(last, js[-1])][1]
                pieces.append(text[cur:start])
                pieces.append(repl)
                cur = end
        pieces.append(text[cur:])
        new = "".join(pieces)
        out[k] = _CTRL_RE.sub("", new)  # belt and braces: no control char survives
    return out, stats
