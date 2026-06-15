# A PubMed-Scale Dataset of Structured Biomedical Abstracts

[![arXiv](https://img.shields.io/badge/arXiv-2606.11361-b31b1b.svg?style=flat)](https://arxiv.org/abs/2606.11361) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20336717.svg)](https://doi.org/10.5281/zenodo.20336717)

## Dataset

The comprehensive structured abstract corpus (23.3M records) is available on the Zenodo [StructuredPubMed](https://doi.org/10.5281/zenodo.20336717).

## Setup for Running Our Scripts

Set up the virtual environment & install the dependencies:
```bash
# Recommend Python 3.11+.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unzip `eval_data.zip` to the project root for the toy samples and pre-computed results.

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
