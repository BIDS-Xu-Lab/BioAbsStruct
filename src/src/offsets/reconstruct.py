#!/usr/bin/env python3
"""Rebuild StructuredPubMed section texts from the offset-only release + PubMed XML.

The release ships no abstract text, only character spans into a reference text that you
rebuild from PubMed yourself:

  Reference text of a PMID: the AbstractText children of MedlineCitation/Article/Abstract
  (OtherAbstract excluded), each "".join(element.itertext()).strip(), empty ones skipped,
  joined with one space. Offsets are Python str (Unicode code point) indices, end exclusive.
  hash = first 16 hex chars of sha256(reference text, UTF-8).

  The offsets were computed on the PubMed 2026 baseline (pubmed26n0001-1334) plus daily
  update files pubmed26n1335-1432 (through 2026-05-02); later files override earlier ones,
  DeleteCitation removes. Records revised in PubMed since then fail the hash check and are
  reported as hash_mismatch instead of being silently mis-sliced.

Input rows (TSV with header):  pmid  section  [label]  spans  hash
  spans = "start-end;start-end"; a section's text = its spans joined with one space.

Usage
  # from local PubMed XML (https://ftp.ncbi.nlm.nih.gov/pubmed/{baseline,updatefiles}/)
  python reconstruct.py --offsets 17M_LLM_labeled_offsets.tsv.gz --xml_dir pubmed/ --out out.tsv
  # a few PMIDs via NCBI E-utilities (current PubMed version)
  python reconstruct.py --offsets 6M_author_provided_offsets.tsv.gz --pmids 18167406 17365800 --efetch --out out.tsv

Output TSV:  pmid  section  [label]  status  text
  status: ok | hash_mismatch | not_found | deleted   (text is empty unless ok)
  --json writes one line per PMID instead: {"pmid", "status", "sections": {section: text}}.

Requires: Python >= 3.8, lxml.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool

csv.field_size_limit(sys.maxsize)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# --- reference text (must stay identical to the release definition) -----------------
def reference_text(abstract_texts) -> str:
    parts = ["".join(el.itertext()).strip() for el in abstract_texts]
    return " ".join(p for p in parts if p)


def ref_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def iter_xml(fileobj, wanted=None):
    """Yield ("article", pmid, reference_text) / ("delete", pmid, None) from a PubMed XML stream."""
    import lxml.etree as ET

    for _, el in ET.iterparse(fileobj, events=("end",), tag=("PubmedArticle", "DeleteCitation"), recover=True):
        if el.tag == "DeleteCitation":
            for p in el.findall("PMID"):
                if p.text and p.text.strip().isdigit():
                    yield "delete", int(p.text), None
        else:
            pm = el.findtext("MedlineCitation/PMID")
            if pm and pm.strip().isdigit() and (wanted is None or int(pm) in wanted):
                yield "article", int(pm), reference_text(el.findall("MedlineCitation/Article/Abstract/AbstractText"))
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]


# --- offsets -------------------------------------------------------------------------
def load_offsets(path: str, pmids: set[int] | None):
    """-> (header, {pmid: [row, ...]})"""
    rows: dict[int, list] = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for row in r:
            pm = int(row[0])
            if pmids is None or pm in pmids:
                rows.setdefault(pm, []).append(row)
    return header, rows


def slice_spans(text: str, spans: str) -> str:
    out = []
    for sp in spans.split(";"):
        s, e = sp.split("-")
        out.append(text[int(s):int(e)])
    return " ".join(out)


class Writer:
    def __init__(self, path: str, header: list[str], as_json: bool):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.as_json = as_json
        self.has_label = "label" in header
        self.counts: dict[str, int] = {}
        if not as_json:
            self.w = csv.writer(self.f, delimiter="\t", lineterminator="\n")
            self.w.writerow(["pmid", "section"] + (["label"] if self.has_label else []) + ["status", "text"])

    def emit(self, pm: int, rows: list, status: str, ref: str | None = None) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        texts = [slice_spans(ref, r[-2]) if status == "ok" else "" for r in rows]
        if self.as_json:
            secs: dict[str, str] = {}
            for r, t in zip(rows, texts):
                secs[r[1]] = (secs[r[1]] + " " + t) if r[1] in secs else t
            self.f.write(json.dumps({"pmid": pm, "status": status, "sections": secs if status == "ok" else {}},
                                    ensure_ascii=False) + "\n")
        else:
            for r, t in zip(rows, texts):
                self.w.writerow([pm, r[1]] + ([r[2]] if self.has_label else []) + [status, t])

    def check(self, pm: int, rows: list, ref: str) -> None:
        self.emit(pm, rows, "ok" if ref_hash(ref) == rows[0][-1] else "hash_mismatch", ref)

    def close(self) -> None:
        self.f.close()


# --- sources -------------------------------------------------------------------------
_WANTED = None


def _parse_file(path):
    with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as f:
        return path, list(iter_xml(f, _WANTED))


def from_xml_dir(xml_dir: str, rows: dict, out: Writer, workers: int) -> None:
    global _WANTED
    files = sorted(glob.glob(os.path.join(xml_dir, "**", "pubmed*.xml*"), recursive=True),
                   key=lambda p: os.path.basename(p))
    if not files:
        sys.exit(f"no pubmed*.xml[.gz] files under {xml_dir}")
    _WANTED = set(rows)
    done: set[int] = set()
    # Newest file first: the first record seen for a PMID is its latest version.
    with Pool(workers) as pool:
        for n, (path, recs) in enumerate(pool.imap(_parse_file, files[::-1], chunksize=1), 1):
            for kind, pm, ref in reversed(recs):
                if pm in done or pm not in rows:
                    continue
                done.add(pm)
                if kind == "delete":
                    out.emit(pm, rows[pm], "deleted")
                else:
                    out.check(pm, rows[pm], ref)
            if n % 100 == 0:
                print(f"  {n}/{len(files)} files, {len(done):,}/{len(rows):,} pmids", file=sys.stderr, flush=True)
    for pm in rows:
        if pm not in done:
            out.emit(pm, rows[pm], "not_found")


def from_efetch(rows: dict, out: Writer, api_key: str | None, batch: int = 200) -> None:
    pmids = sorted(rows)
    pause = 0.11 if api_key else 0.35  # NCBI limits: 10 req/s with a key, 3 without
    seen: set[int] = set()
    for i in range(0, len(pmids), batch):
        q = {"db": "pubmed", "retmode": "xml", "id": ",".join(map(str, pmids[i:i + batch]))}
        if api_key:
            q["api_key"] = api_key
        data = urllib.request.urlopen(EUTILS, urllib.parse.urlencode(q).encode()).read()
        for kind, pm, ref in iter_xml(io.BytesIO(data)):
            if kind == "article" and pm in rows and pm not in seen:
                seen.add(pm)
                out.check(pm, rows[pm], ref)
        time.sleep(pause)
    for pm in pmids:
        if pm not in seen:
            out.emit(pm, rows[pm], "not_found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", required=True, help="release offsets TSV")
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml_dir", help="directory with PubMed baseline/updatefiles XML (.xml or .xml.gz)")
    src.add_argument("--efetch", action="store_true", help="fetch from NCBI E-utilities (current versions)")
    ap.add_argument("--pmids", nargs="*", type=int, help="only these PMIDs")
    ap.add_argument("--pmid_file", help="only PMIDs listed in this file (one per line)")
    ap.add_argument("--api_key", default=os.environ.get("NCBI_API_KEY"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json", action="store_true", help="one JSON line per PMID instead of TSV rows")
    a = ap.parse_args()

    want = set(a.pmids or [])
    if a.pmid_file:
        want |= {int(x) for x in open(a.pmid_file) if x.strip()}
    header, rows = load_offsets(a.offsets, want or None)
    print(f"{len(rows):,} PMIDs to rebuild", file=sys.stderr)
    out = Writer(a.out, header, a.json)
    if a.efetch:
        from_efetch(rows, out, a.api_key)
    else:
        from_xml_dir(a.xml_dir, rows, out, a.workers)
    out.close()
    print(json.dumps(out.counts), file=sys.stderr)


if __name__ == "__main__":
    main()
