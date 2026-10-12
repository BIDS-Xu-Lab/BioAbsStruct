"""Offset-only (CC0) release: section labels as character spans into a PubMed reference text.

Reference text (the only text the offsets point into; users rebuild it with reconstruct.py):
  PubMed 2026 baseline pubmed26n0001-1334 + updatefiles pubmed26n1335-1432 (to 2026-05-02),
  later files override earlier ones, DeleteCitation removes. Per PubmedArticle, the
  AbstractText children of MedlineCitation/Article/Abstract (OtherAbstract excluded), each
  "".join(el.itertext()).strip(), empty ones skipped, joined with a single space.
  Offsets are Python str (Unicode code point) indices; hash = sha256(utf-8)[:16].

Row format (one row per pmid x section):  pmid  section  [label]  spans  hash
  spans = "start-end;start-end" (end exclusive, ascending).

Subcommands (shards are pmid % 32, aligned with fidelity_flags/shards):
  prep    stream the 23M TSV and the 6M release ids, parse the XML -> <work>/ref/NNN.tsv, <work>/abs/NNN.tsv
  shard   one SLURM array task: step-0 comparison + 6M rows + 17M rows for shard NNN
  merge   concatenate shards -> final files + summary json
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

csv.field_size_limit(sys.maxsize)
N_SHARDS = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- reference text ------------------------------------------------------------------
def ref_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_reference(elements: list[list[str]]) -> tuple[str, list[tuple[int, int]]]:
    """elements = [[label, nlm, text], ...] -> (reference text, span of each element)."""
    parts, spans, pos = [], [], 0
    for _, _, t in elements:
        if parts:
            pos += 1
        spans.append((pos, pos + len(t)))
        parts.append(t)
        pos += len(t)
    return " ".join(parts), spans


def parse_xml_file(path: str):
    """-> (path, [(pmid, elements_json)], [deleted pmids]). Elements with empty text dropped."""
    import lxml.etree as ET

    rows, deleted = [], []
    with gzip.open(path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",), tag=("PubmedArticle", "DeleteCitation"), recover=True):
            if el.tag == "DeleteCitation":
                deleted.extend(int(p.text) for p in el.findall("PMID") if p.text and p.text.strip().isdigit())
            else:
                pm = el.findtext("MedlineCitation/PMID")
                if pm and pm.strip().isdigit():
                    elems = []
                    for at in el.findall("MedlineCitation/Article/Abstract/AbstractText"):
                        t = "".join(at.itertext()).strip()
                        if t:
                            elems.append([(at.get("Label") or "").strip(), (at.get("NlmCategory") or "").strip(), t])
                    rows.append((int(pm), json.dumps(elems, ensure_ascii=False)))
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    return path, rows, deleted


def xml_files(ftp: str) -> list[str]:
    return sorted(glob.glob(os.path.join(ftp, "baseline", "*.xml.gz"))) + sorted(
        glob.glob(os.path.join(ftp, "updatefiles", "*.xml.gz")))


# --- prep ----------------------------------------------------------------------------
def cmd_prep(a) -> None:
    import pandas as pd

    work = Path(a.work_dir)
    for d in ("abs", "ref", "parts"):
        (work / d).mkdir(parents=True, exist_ok=True)

    # 6M release pmids
    ids6 = pd.read_csv(f"{a.data_dir}/6M_author_provided_subset.tsv", sep="\t", usecols=["pmid"], dtype={"pmid": np.int64})
    np.save(work / "release_6M_pmids.npy", np.sort(ids6["pmid"].to_numpy()))
    log(f"6M release pmids: {len(ids6):,}")
    del ids6

    # 23M abstract column, sharded
    outs = [open(work / "abs" / f"{k:03d}.tsv", "w", newline="", encoding="utf-8") for k in range(N_SHARDS)]
    ws = [csv.writer(f, delimiter="\t") for f in outs]
    pmids = []
    for ci, ch in enumerate(pd.read_csv(f"{a.data_dir}/23M_research_type_records_with_abstracts.tsv", sep="\t",
                                        usecols=["pmid", "contain_structured", "abstract"],
                                        dtype={"pmid": np.int64, "abstract": str, "contain_structured": str},
                                        chunksize=500_000)):
        for pm, cs, ab in zip(ch["pmid"], ch["contain_structured"], ch["abstract"]):
            ws[pm % N_SHARDS].writerow([pm, cs, "" if pd.isna(ab) else ab])
        pmids.append(ch["pmid"].to_numpy())
        if (ci + 1) % 10 == 0:
            log(f"  23M chunks {ci + 1}")
    for f in outs:
        f.close()
    want = np.unique(np.concatenate(pmids))
    np.save(work / "pmids_23M.npy", want)
    log(f"23M pmids: {want.size:,}")

    # XML: parse in parallel, keep only 23M pmids, one part file per XML file
    files = xml_files(a.ftp_dir)
    log(f"{len(files)} xml files")
    global _WANT
    _WANT = want
    latest: dict[int, int] = {}
    deleted_after: dict[int, int] = {}
    with Pool(a.workers) as pool:
        for n, (fi, nrows, dels) in enumerate(pool.imap(_parse_to_part, list(enumerate(files)), chunksize=1), 1):
            for pm in np.load(work / "parts" / f"{fi:04d}.pmids.npy"):
                latest[int(pm)] = fi
            for pm in dels:
                deleted_after[pm] = fi
            if n % 100 == 0:
                log(f"  parsed {n}/{len(files)}  pmids={len(latest):,}")
    gone = {pm for pm, fi in deleted_after.items() if pm in latest and fi > latest[pm]}
    log(f"23M pmids found in XML: {len(latest):,}; deleted later: {len(gone):,}")

    outs = [open(work / "ref" / f"{k:03d}.tsv", "w", newline="", encoding="utf-8") for k in range(N_SHARDS)]
    ws = [csv.writer(f, delimiter="\t") for f in outs]
    for fi in range(len(files)):
        with open(work / "parts" / f"{fi:04d}.tsv", newline="", encoding="utf-8") as f:
            for pm, elems in csv.reader(f, delimiter="\t"):
                pm = int(pm)
                if latest.get(pm) == fi and pm not in gone:
                    ws[pm % N_SHARDS].writerow([pm, elems, os.path.basename(files[fi])])
    for f in outs:
        f.close()
    json.dump({"xml_files": len(files), "first": os.path.basename(files[0]), "last": os.path.basename(files[-1]),
               "pmids_23M": int(want.size), "found_in_xml": len(latest), "deleted_later": len(gone)},
              open(work / "prep_summary.json", "w"), indent=1)
    log("prep done")


_WANT = None


def _parse_to_part(item):
    fi, path = item
    out = Path(WORK_DIR) / "parts"
    _, rows, dels = parse_xml_file(path)
    pm = np.array([r[0] for r in rows], dtype=np.int64)
    keep = np.isin(pm, _WANT)
    with open(out / f"{fi:04d}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r, k in zip(rows, keep):
            if k:
                w.writerow(r)
    np.save(out / f"{fi:04d}.pmids.npy", pm[keep])
    return fi, int(keep.sum()), dels


WORK_DIR = None


# --- per-shard -----------------------------------------------------------------------
def spans_str(sp: list[tuple[int, int]]) -> str:
    return ";".join(f"{s}-{e}" for s, e in sp)


def merge_adjacent(sp: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Merge spans separated only by whitespace (consecutive sentences of one section)."""
    out = []
    for s, e in sorted(sp):
        if out and not text[out[-1][1]:s].strip():
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def rows_6m(pmid: int, elements: list, ref: str, el_spans: list, h: str) -> list[list]:
    """One row per (NlmCategory, Label); section = PubMed's NlmCategory as given, NONE if absent."""
    groups: dict[tuple[str, str], list] = {}
    for (lab, nlm, _), sp in zip(elements, el_spans):
        sec = nlm or "NONE"
        groups.setdefault((sec, lab), []).append(sp)
    rows = [[pmid, sec, lab, spans_str(sp), h] for (sec, lab), sp in groups.items()]
    return sorted(rows, key=lambda r: int(r[3].split("-", 1)[0]))


def label_17m(ref: str, sections: dict[str, str]):
    """-> (sentence spans, label per sentence or None, stats). Same alignment as the fidelity
    check (normalized, fuzzy split/merge); a source sentence spread over several sections
    takes the section holding most of its characters."""
    from sentence_matching import align, labeled_prediction_sentences, split_sentence_spans

    sp = split_sentence_spans(ref)
    src = [ref[s:e] for s, e in sp]
    labeled = labeled_prediction_sentences(sections)
    pred = [s for s, _ in labeled]
    al = align(src, pred, "normalized", fuzzy_resegment=True)
    lab: list[str | None] = [None] * len(src)
    for j, i in enumerate(al.pred_to_src):
        if i is not None:
            lab[i] = labeled[j][1]
    for i, j, _ in al.modified:
        lab[i] = labeled[j][1]
    cross = 0
    groups = [(s, p) for s, p in al.resegmented] + [(s, p) for s, p, _ in al.split_merge]
    for srcs, preds in groups:
        w = Counter()
        for j in preds:
            w[labeled[j][1]] += len(pred[j])
        if len(w) > 1:
            cross += len(srcs)
        best = max(w.items(), key=lambda kv: (kv[1], -SECTION_IDX[kv[0]]))[0]
        for i in srcs:
            lab[i] = best
    # a duplicated pred sentence in another section than its source's label
    by_key: dict[str, set] = defaultdict(set)
    for i, s in enumerate(src):
        by_key[src_key(s)].add(lab[i])
    dup_x = sum(1 for j in al.duplicated_pred if labeled[j][1] not in by_key[src_key(pred[j])])
    stats = {"n_sent": len(src), "n_none": lab.count(None), "n_cross_split": cross, "n_dup_other_section": dup_x,
             "n_added": len(al.added_pred), "reordered": int(al.n_moved > 0)}
    return sp, lab, stats


def src_key(s: str) -> str:
    from sentence_matching import sentence_key
    return sentence_key(s, "normalized")


SECTION_IDX = {k: i for i, k in enumerate(["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"])}


def rows_17m(pmid: int, ref: str, sp, lab, h: str) -> list[list]:
    by: dict[str, list] = defaultdict(list)
    for (s, e), l in zip(sp, lab):
        by[l or "NONE"].append((s, e))
    rows = [[pmid, sec, spans_str(merge_adjacent(v, ref)), h] for sec, v in by.items()]
    return sorted(rows, key=lambda r: int(r[2].split("-", 1)[0]))


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def _one_17m(item):
    from sentence_matching import parse_prediction

    pm, pred, el = item
    sections = parse_prediction(pred)
    if not sections:
        return pm, "all_null", None, None
    if not el:
        return pm, "no_xml_text", None, None
    r, _ = build_reference(el)
    try:
        sp, lab, st = label_17m(r, sections)
    except Exception as e:  # noqa: BLE001
        return pm, f"error:{type(e).__name__}", None, None
    return pm, "ok", rows_17m(pm, r, sp, lab, ref_hash(r)), st


def cmd_shard(a) -> None:
    k = a.shard if a.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    work = Path(a.work_dir)
    for d in ("step0", "out6", "out17"):
        (work / d).mkdir(parents=True, exist_ok=True)
    rel6 = np.load(work / "release_6M_pmids.npy")

    ref: dict[int, list] = {}
    with open(work / "ref" / f"{k:03d}.tsv", newline="", encoding="utf-8") as f:
        for pm, el, src_file in csv.reader(f, delimiter="\t"):
            ref[int(pm)] = json.loads(el)
    log(f"shard {k}: {len(ref):,} reference records")

    # step 0: does the reference text reproduce the 23M `abstract` column?
    c, ex = Counter(), []
    with open(work / "abs" / f"{k:03d}.tsv", newline="", encoding="utf-8") as f:
        for pm, cs, ab in csv.reader(f, delimiter="\t"):
            pm = int(pm)
            grp = "structured" if cs == "True" else "unstructured"
            el = ref.get(pm)
            if el is None:
                c[f"{grp}:no_xml"] += 1
                continue
            r, _ = build_reference(el)
            withlab = " ".join(f"{l} {t}" if l else t for l, _, t in el)
            ab = ab.strip()
            if ab == r:
                m = "exact"
            elif ab == withlab:
                m = "exact_with_labels"
            elif norm_ws(ab) == norm_ws(r):
                m = "ws_only"
            elif norm_ws(ab) == norm_ws(withlab):
                m = "ws_only_with_labels"
            else:
                m = "differs"
                if len(ex) < 30:
                    ex.append({"pmid": pm, "abstract": ab[:400], "ref_with_labels": withlab[:400]})
            c[f"{grp}:{m}"] += 1
    json.dump({"counts": dict(c), "examples": ex}, open(work / "step0" / f"{k:03d}.json", "w"), indent=1,
              ensure_ascii=False)
    log(f"step0: {dict(c)}")

    # 6M
    n6 = Counter()
    with open(work / "out6" / f"{k:03d}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for pm in rel6[rel6 % N_SHARDS == k]:
            pm = int(pm)
            el = ref.get(pm)
            if not el:
                n6["no_xml_text"] += 1
                continue
            r, esp = build_reference(el)
            w.writerows(rows_6m(pm, el, r, esp, ref_hash(r)))
            n6["ok"] += 1
    log(f"6M: {dict(n6)}")

    if a.skip_17m:
        return
    # 17M (predictions = encfixed, from fidelity_flags shards)
    n17, agg = Counter(), Counter()
    t0 = time.time()
    with open(f"{a.flag_shards_dir}/{k:03d}.pred.tsv", newline="", encoding="utf-8") as fi, \
            open(work / "out17" / f"{k:03d}.tsv", "w", newline="", encoding="utf-8") as fo, \
            open(work / "out17" / f"{k:03d}.stats.tsv", "w", newline="", encoding="utf-8") as fs, \
            open(work / "out17" / f"{k:03d}.excluded.tsv", "w", newline="", encoding="utf-8") as fx:
        w, ws_, wx = (csv.writer(x, delimiter="\t") for x in (fo, fs, fx))
        def items():
            for n, (pm, pred, *_rest) in enumerate(csv.reader(fi, delimiter="\t"), 1):
                if a.limit and n > a.limit:
                    break
                pm = int(pm)
                yield pm, pred, ref.get(pm)

        with Pool(a.workers) as pool:
            for n, (pm, status, rows, st) in enumerate(pool.imap(_one_17m, items(), chunksize=200), 1):
                n17[status] += 1
                if status != "ok":
                    wx.writerow([pm, status])
                    continue
                w.writerows(rows)
                ws_.writerow([pm, st["n_sent"], st["n_none"], st["n_cross_split"], st["n_dup_other_section"]])
                agg.update(st)
                agg["rec_with_none"] += st["n_none"] > 0
                agg["rec_with_cross_split"] += st["n_cross_split"] > 0
                agg["rec_with_dup_other_section"] += st["n_dup_other_section"] > 0
                if n % 50_000 == 0:
                    log(f"  17M {n:,} ({time.time() - t0:.0f}s)")
    json.dump({"status6": dict(n6), "status17": dict(n17), "agg17": dict(agg)},
              open(work / "out17" / f"{k:03d}.summary.json", "w"), indent=1)
    log(f"17M: {dict(n17)}  {dict(agg)}")


# --- merge ---------------------------------------------------------------------------
def cmd_merge(a) -> None:
    work = Path(a.work_dir)
    final = work / "final"
    final.mkdir(exist_ok=True)
    s0 = Counter()
    ex = []
    for p in sorted(glob.glob(str(work / "step0" / "*.json"))):
        d = json.load(open(p))
        s0.update(d["counts"])
        ex += d["examples"][:3]
    for name, hdr, sub in (("6M_author_provided_offsets.tsv", "pmid\tsection\tlabel\tspans\thash\n", "out6/{:03d}.tsv"),
                           ("17M_LLM_labeled_offsets.tsv", "pmid\tsection\tspans\thash\n", "out17/{:03d}.tsv"),
                           ("17M_LLM_labeled_offsets_stats.tsv", "pmid\tn_sent\tn_none\tn_cross_split\tn_dup_other_section\n",
                            "out17/{:03d}.stats.tsv"),
                           ("17M_excluded_pmids.tsv", "pmid\treason\n", "out17/{:03d}.excluded.tsv")):
        with open(final / name, "w", encoding="utf-8") as fo:
            fo.write(hdr)
            for k in range(N_SHARDS):
                with open(work / sub.format(k), encoding="utf-8") as fi:
                    for line in fi:
                        fo.write(line)
        log(f"wrote {name}")
    st6, st17, agg = Counter(), Counter(), Counter()
    for k in range(N_SHARDS):
        d = json.load(open(work / "out17" / f"{k:03d}.summary.json"))
        st6.update(d["status6"]); st17.update(d["status17"]); agg.update(d["agg17"])
    json.dump({"step0": dict(s0), "step0_examples": ex, "status6": dict(st6), "status17": dict(st17), "agg17": dict(agg),
               "prep": json.load(open(work / "prep_summary.json"))},
              open(final / "summary.json", "w"), indent=1, ensure_ascii=False)
    log(f"step0 {dict(s0)}\n6M {dict(st6)}\n17M {dict(st17)}\nagg {dict(agg)}")


def main():
    global WORK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "shard", "merge"])
    ap.add_argument("--work_dir", required=True, help="intermediate + final outputs (<work_dir>/final/)")
    ap.add_argument("--ftp_dir", help="prep: PubMed baseline/updatefiles XML dir (pubmed*.xml[.gz])")
    ap.add_argument("--data_dir", help="prep: dir with 6M_author_provided_subset.tsv and 23M_research_type_records_with_abstracts.tsv")
    ap.add_argument("--flag_shards_dir", help="shard (17M): fidelity_flags shards dir with NNN.pred.tsv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_17m", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prep" and not (a.ftp_dir and a.data_dir):
        ap.error("prep needs --ftp_dir and --data_dir")
    if a.cmd == "shard" and not a.skip_17m and not a.flag_shards_dir:
        ap.error("shard needs --flag_shards_dir (or --skip_17m)")
    WORK_DIR = a.work_dir
    {"prep": cmd_prep, "shard": cmd_shard, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
