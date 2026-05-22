#!/usr/bin/env bash
# Reproduce raw metrics for Figure 3 and Table 3 (10K samples × three article types × three models).
# Usage: bash scripts/reproduce_paper_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for article_type in 10K_research_articles 10K_randomized_clinical_trials 10K_observational_studies; do
  echo "=============================================="
  echo "Evaluating Article Type: ${article_type}"
  echo "=============================================="
  for model in BERT_CLS GPT-4.1-nano GPT-4.1-mini; do
    echo "---------- Model: ${model} ----------"
    python src/eval.py \
      "eval_data/${article_type}/10K_references.tsv" \
      "eval_data/${article_type}/10K_${model}_results.tsv"
    echo ""
  done
  echo ""
done
