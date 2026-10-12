# StructuredPubMed — offset-only release (CC0)

This release contains **no abstract text**. Each row points to character spans in a
*reference text* that you rebuild from PubMed with `reconstruct.py`.

## Files

| File | Rows | Columns |
|---|---|---|
| `6M_author_provided_offsets.tsv.gz` | one per PMID x (NlmCategory, Label) | `pmid  section  label  spans  hash` |
| `17M_LLM_labeled_offsets.tsv.gz` | one per PMID x section | `pmid  section  spans  hash` |
| `17M_LLM_labeled_offsets_stats.tsv.gz` | one per PMID | `pmid  n_sent  n_none  n_cross_split  n_dup_other_section` |
| `6M_excluded_pmids.tsv.gz`, `17M_excluded_pmids.tsv.gz` | one per PMID | `pmid  reason` (`all_null`: model returned no sections; `no_xml_text`: no abstract text in the snapshot, e.g. deleted from PubMed) |
| `reconstruct.py` | | rebuilds section texts (Python >= 3.8, `lxml`) |

- `spans`: `start-end;start-end;...` — 0-based, end-exclusive Python `str` (Unicode code point)
  offsets into the reference text, ascending. A section's text is its spans joined with one space.
- `hash`: first 16 hex characters of SHA-256 of the UTF-8 reference text. `reconstruct.py`
  refuses to slice a record whose hash differs (`status=hash_mismatch`).

## Reference text

- Source: PubMed 2026 baseline `pubmed26n0001`–`pubmed26n1334` plus update files
  `pubmed26n1335`–`pubmed26n1432` (through 2026-05-02), downloaded 2026-05-03 from
  <https://ftp.ncbi.nlm.nih.gov/pubmed/>. A later file overrides an earlier one for the same
  PMID; `DeleteCitation` removes it.
- Per `PubmedArticle`: the `AbstractText` children of `MedlineCitation/Article/Abstract`
  (`OtherAbstract` excluded), in document order; each element's text is
  `"".join(element.itertext()).strip()` (inline markup such as `<i>`, `<sup>` contributes its
  text); empty elements are skipped; elements are joined with a single space.
  Section headings (`Label`) are **not** part of the text.

## Sections

**6M (author-provided).** One span per `AbstractText` element, exact by construction.
`section` is PubMed's `NlmCategory` as given (BACKGROUND, OBJECTIVE, METHODS, RESULTS,
CONCLUSIONS, UNASSIGNED), or `NONE` when the element has no NlmCategory; `label` is the
author's original heading (e.g. `INTRODUCTION`, `PURPOSE`), empty if none.

**17M (LLM-labeled).** GPT-4.1-mini rewrote each unstructured abstract into five sections.
The reference text was split into sentences (pysbd 0.3.4 plus the project wrapper) and each
sentence was aligned to the model output; it takes the section the model put it in.
`section` is one of the five labels or `NONE`:

- `NONE` — the sentence did not appear in the model output (omitted by the model).
- Sentence split by the model across two sections → assigned to the section that holds most of
  its characters (counted in `n_cross_split`).
- Sentence emitted twice in different sections → first occurrence (counted in `n_dup_other_section`).
- Reordered sentences → spans follow the source order; a section may have several spans.
- Text the model added or reworded is **not** represented: the offsets carry the original
  sentences under the model's labels, not the model's wording.
- Records where the model returned no sections (`all_null`) or that have no abstract text in the
  snapshot are listed in `17M_excluded_pmids.tsv.gz`.

## Rebuilding

```bash
pip install lxml
# Download the XML files listed above into pubmed/ (baseline/ and updatefiles/), then
python reconstruct.py --offsets 17M_LLM_labeled_offsets.tsv.gz --xml_dir pubmed/ --out 17M_text.tsv
# or a handful of PMIDs from NCBI E-utilities (current PubMed version)
python reconstruct.py --offsets 6M_author_provided_offsets.tsv.gz --pmids 18167406 --efetch --out x.tsv
```

Output: `pmid  section  [label]  status  text` (or `--json`: one line per PMID). `status` is
`ok`, `hash_mismatch` (record changed in PubMed since the snapshot), `not_found`, or `deleted`.

NLM replaces the baseline every year. If the 2026 files are no longer on the FTP server,
use a later baseline: unchanged records rebuild normally and revised ones are reported as
`hash_mismatch`.

## License

The offset files are released under CC0 1.0. The abstract text belongs to its publishers
and is obtained from PubMed under NLM's terms.
