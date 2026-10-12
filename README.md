# A dataset of section-labeled biomedical abstracts from PubMed

[![arXiv](https://img.shields.io/badge/arXiv-2606.11361-b31b1b.svg?style=flat)](https://arxiv.org/abs/2606.11361) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23162488.svg)](https://doi.org/10.5281/zenodo.23162488)

## Dataset

The dataset, [BioAbsStruct](https://doi.org/10.5281/zenodo.23162488), is available on Zenodo under the CC0 1.0 license. It contains 5,993,607 records with author-provided section labels and 17,276,110 records labeled by GPT-4.1-mini.

The release contains no abstract text. For each record, it specifies which character spans of the PubMed abstract belong to each section, together with a hash of the abstract text. Users obtain the abstracts from the PubMed XML files distributed by NLM and reconstruct the section text with the script below.

## Setup for Running Our Scripts

Set up the virtual environment & install the dependencies:
```bash
# Recommend Python 3.11+.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unzip `eval_data.zip` to the project root for the toy samples and pre-computed results.

## Reconstruct Section Text from the Released Offsets

Download the offset files from Zenodo and the PubMed baseline and update XML files from https://ftp.ncbi.nlm.nih.gov/pubmed/, then run:

```bash
# Full subset from a local copy of the PubMed XML files
python src/offsets/reconstruct.py --offsets 17M_LLM_labeled_offsets.tsv.gz --xml_dir pubmed/ --out 17M_text.tsv

# A few records retrieved from NCBI E-utilities
python src/offsets/reconstruct.py --offsets 6M_author_provided_offsets.tsv.gz --pmids 18167406 17365800 --efetch --out 6M_sample.tsv
```

The script checks each abstract against the released hash. Records revised in PubMed after our May 2026 snapshot are reported as `hash_mismatch` and are not reconstructed. See `src/offsets/README_offsets.md` for details.

## Extract Structured Abstracts from PubMed XML Dump

Set `PROJECT_DATA` to a folder with enough disk space, then run:

```bash
export PROJECT_DATA=YOUR_DATA_FOLDER_PATH
bash scripts/extract_structured_abstracts_from_pubmed.sh
```

This downloads the latest PubMed baseline and updatefiles from NCBI FTP, parses structured abstracts from the XML, and writes `integrated_structured_abstracts.tsv` under `PROJECT_DATA`.

## Abstract Structuring via GPT
Create `.env` in the current project folder and configure endpoint and api key there:
```bash
# OpenAI-compatible base URL
endpoint=ENDPOINT_URL
api_key=YOUR_API_KEY
``` 

Run the following command to extract structured abstracts from 10 toy samples:
```bash
python src/extract_structured_abstracts_via_GPT.py --input_tsv eval_data/10_toy_samples.tsv --output_tsv eval_data/10_toy_samples_GPT-4.1-mini_predicted_structured_abstract.tsv --deployment_name gpt-4.1-mini
```

Please note that to handle million-scale data we use Batch API to speed up the extraction process.

## Reproduce Technical Validation Results

Run the evaluation script on the toy samples:
```bash
python src/eval.py eval_data/10_toy_samples.tsv eval_data/10_toy_samples_GPT-4.1-mini_predicted_structured_abstract.tsv
```

Reproduce raw data for Figure 3 and Table 3 reported in the paper:

```bash
bash scripts/reproduce_paper_eval.sh
```

## Citation

If you use our dataset/code in your research, please cite our preprint:

```bibtex
@article{chang2026pubmed,
  title={A PubMed-Scale Dataset of Structured Biomedical Abstracts},
  author={Chang, Chia-Hsuan and Song, Haerin and Ondov, Brian and Xu, Hua},
  journal={arXiv preprint arXiv:2606.11361},
  year={2026}
}
```
