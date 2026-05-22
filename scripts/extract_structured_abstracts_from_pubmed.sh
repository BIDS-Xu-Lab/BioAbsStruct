#!/usr/bin/env bash
# Download PubMed baseline/update XML and build integrated_structured_abstracts.tsv.
# Usage: export PROJECT_DATA=/path/to/data && bash scripts/extract_structured_abstracts_from_pubmed.sh

set -euo pipefail

if [[ -z "${PROJECT_DATA:-}" ]]; then
  echo "Error: set PROJECT_DATA to your data folder path, e.g. export PROJECT_DATA=/path/to/data"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$PROJECT_DATA/baseline" "$PROJECT_DATA/updatefiles"

echo "Downloading baseline XML from NCBI FTP..."
(cd "$PROJECT_DATA/baseline" && wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.xml.gz)

echo "Downloading updatefiles XML from NCBI FTP..."
(cd "$PROJECT_DATA/updatefiles" && wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/*.xml.gz)

echo "Parsing baseline XML..."
python src/extract_structured_abstracts_from_pubmed_xml.py \
  --log_file_path "$PROJECT_DATA/baseline_parsing.log" \
  --xml_gz_file_folder "$PROJECT_DATA/baseline" \
  --folder_out "$PROJECT_DATA/baseline_parsed_dfs"

echo "Parsing updatefiles XML..."
python src/extract_structured_abstracts_from_pubmed_xml.py \
  --log_file_path "$PROJECT_DATA/updatefiles_parsing.log" \
  --xml_gz_file_folder "$PROJECT_DATA/updatefiles" \
  --folder_out "$PROJECT_DATA/updatefiles_parsed_dfs"

echo "Merging parsed results..."
python src/merge_baseline_and_update_tsvs.py \
  --baseline_dir "$PROJECT_DATA/baseline_parsed_dfs" \
  --updatefiles_dir "$PROJECT_DATA/updatefiles_parsed_dfs" \
  --output_tsv "$PROJECT_DATA/integrated_structured_abstracts.tsv"

echo "Done. Output: $PROJECT_DATA/integrated_structured_abstracts.tsv"
