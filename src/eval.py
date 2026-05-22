"""
Evaluate predicted structured abstracts against gold references.

Reads two TSVs (gold with structured_abstract JSON, predictions with
predicted_structured_abstract JSON), aligns rows by pmid, and reports
ROUGE-1/2/L (F1) and corpus-style sentence BLEU averaged per section key.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from typing import Any

import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


SECTION_KEYS = (
    "BACKGROUND",
    "OBJECTIVE",
    "METHODS",
    "RESULTS",
    "CONCLUSIONS",
)


def _parse_json_dict(cell: Any) -> dict[str, str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    s = str(cell).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in obj.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[str(k)] = ""
        else:
            out[str(k)] = str(v).strip()
    return out


def _dedupe_pmid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    dup = df["pmid"].duplicated(keep=False)
    if dup.any():
        n = df.loc[dup, "pmid"].nunique()
        print(
            f"Warning: {label} has duplicate pmid rows ({n} pmids); keeping last row each.",
            file=sys.stderr,
        )
    return df.drop_duplicates(subset=["pmid"], keep="last")


def load_gold_pred(
    gold_path: str,
    pred_path: str,
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    gold_df = pd.read_csv(gold_path, sep="\t", dtype={"pmid": str})
    pred_df = pd.read_csv(pred_path, sep="\t", dtype={"pmid": str})

    if "pmid" not in gold_df.columns:
        raise SystemExit(f"Gold TSV missing required column 'pmid': {gold_path}")
    if "structured_abstract" not in gold_df.columns:
        raise SystemExit(
            f"Gold TSV missing required column 'structured_abstract': {gold_path}"
        )
    if "pmid" not in pred_df.columns:
        raise SystemExit(f"Prediction TSV missing required column 'pmid': {pred_path}")
    if "predicted_structured_abstract" not in pred_df.columns:
        raise SystemExit(
            f"Prediction TSV missing required column "
            f"'predicted_structured_abstract': {pred_path}"
        )

    gold_df = _dedupe_pmid(gold_df, "gold")
    pred_df = _dedupe_pmid(pred_df, "predictions")

    gold_map = {
        str(r["pmid"]): _parse_json_dict(r["structured_abstract"])
        for _, r in gold_df.iterrows()
    }
    pred_map = {
        str(r["pmid"]): _parse_json_dict(r["predicted_structured_abstract"])
        for _, r in pred_df.iterrows()
    }

    common = sorted(set(gold_map) & set(pred_map))
    only_gold = set(gold_map) - set(pred_map)
    only_pred = set(pred_map) - set(gold_map)
    if only_gold:
        print(
            f"Warning: {len(only_gold)} pmids only in gold TSV (skipped).",
            file=sys.stderr,
        )
    if only_pred:
        print(
            f"Warning: {len(only_pred)} pmids only in prediction TSV (skipped).",
            file=sys.stderr,
        )

    rows: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for pmid in common:
        rows.append((pmid, gold_map[pmid], pred_map[pmid]))
    return rows


def evaluate_key_level(
    aligned: list[tuple[str, dict[str, str], dict[str, str]]],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    bleu_metric = BLEU(effective_order=True)

    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0, "bleu": 0.0}
    )
    counts: dict[str, int] = defaultdict(int)

    for _pmid, gold, pred in aligned:
        for key in SECTION_KEYS:
            ref = (gold.get(key) or "").strip()
            if not ref:
                continue
            hyp = (pred.get(key) or "").strip()
            counts[key] += 1

            r = rouge.score(ref, hyp)
            sums[key]["rouge1_f"] += float(r["rouge1"].fmeasure)
            sums[key]["rouge2_f"] += float(r["rouge2"].fmeasure)
            sums[key]["rougeL_f"] += float(r["rougeL"].fmeasure)

            bleu_score = bleu_metric.sentence_score(hyp, [ref]).score
            sums[key]["bleu"] += float(bleu_score)

    means: dict[str, dict[str, float]] = {}
    for key in SECTION_KEYS:
        n = counts[key]
        if n == 0:
            means[key] = {
                "rouge1_f": float("nan"),
                "rouge2_f": float("nan"),
                "rougeL_f": float("nan"),
                "bleu": float("nan"),
            }
            continue
        means[key] = {m: sums[key][m] / n for m in sums[key]}

    return means, dict(counts)


def mean_std_across_section_keys(
    means: dict[str, dict[str, float]],
    counts: dict[str, int],
) -> dict[str, tuple[float, float]]:
    """Mean and sample stdev of per-section means, one pair per metric."""
    metrics = ("rouge1_f", "rouge2_f", "rougeL_f", "bleu")
    out: dict[str, tuple[float, float]] = {}
    for mk in metrics:
        vals = [
            means[k][mk]
            for k in SECTION_KEYS
            if counts.get(k, 0) > 0 and not math.isnan(means[k][mk])
        ]
        if not vals:
            out[mk] = (float("nan"), float("nan"))
        elif len(vals) == 1:
            out[mk] = (vals[0], float("nan"))
        else:
            out[mk] = (statistics.mean(vals), statistics.stdev(vals))
    return out


def _fmt_mean_std(mu: float, sigma: float) -> str:
    if math.isnan(mu):
        return f"{'nan':>8}"
    if math.isnan(sigma):
        return f"{mu:8.4f}"
    return f"{mu:8.4f}±{sigma:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ROUGE-1/2/L (F1) and sentence BLEU per section key "
            "(structured_abstract vs predicted_structured_abstract)."
        )
    )
    parser.add_argument(
        "gold_tsv",
        help="TSV with columns pmid and structured_abstract (JSON object).",
    )
    parser.add_argument(
        "pred_tsv",
        help="TSV with columns pmid and predicted_structured_abstract (JSON object).",
    )
    args = parser.parse_args()

    aligned = load_gold_pred(args.gold_tsv, args.pred_tsv)
    if not aligned:
        raise SystemExit("No overlapping pmids between gold and prediction TSVs.")

    means, counts = evaluate_key_level(aligned)

    print(f"Aligned PMIDs: {len(aligned)}\n")
    header = (
        f"{'section':<14}\t{'n':>5}\t"
        f"{'ROUGE-1':>8}\t{'ROUGE-2':>8}\t{'ROUGE-L':>8}\t{'BLEU':>8}"
    )
    print(header)
    print("-" * len(header.expandtabs()))
    for key in SECTION_KEYS:
        n = counts.get(key, 0)
        m = means[key]
        print(
            f"{key:<14}\t{n:5d}\t"
            f"{m['rouge1_f']:8.4f}\t{m['rouge2_f']:8.4f}\t"
            f"{m['rougeL_f']:8.4f}\t{m['bleu']:8.4f}"
        )

    agg = mean_std_across_section_keys(means, counts)
    print("-" * len(header.expandtabs()))
    r1_m, r1_s = agg["rouge1_f"]
    r2_m, r2_s = agg["rouge2_f"]
    rl_m, rl_s = agg["rougeL_f"]
    bl_m, bl_s = agg["bleu"]
    print(
        f"{'mean±std (keys)':<14}\t{'—':>5}\t"
        f"{_fmt_mean_std(r1_m, r1_s)}\t{_fmt_mean_std(r2_m, r2_s)}\t"
        f"{_fmt_mean_std(rl_m, rl_s)}\t{_fmt_mean_std(bl_m, bl_s)}"
    )


if __name__ == "__main__":
    main()
