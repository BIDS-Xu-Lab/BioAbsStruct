"""
Merge parsed PubMed baseline TSVs with updatefile TSVs (NLM load order).

Follows NLM guidance: build the reference from all baseline chunk files, then
apply each daily update chunk in sorted filename order. For a given PMID, the
last-seen row wins (revised citations replace earlier rows).

Deletion caveat: update XML can list deleted citations (DeleteCitation). The
structured-abstract extractor only emits PubmedArticle rows, so this script
only upserts by PMID. PMIDs removed from PubMed may still appear in the output
until deletes are represented in the parsed tables or handled elsewhere.

Large runs require enough RAM to hold the integrated DataFrame; for very large
corpora consider an out-of-core tool instead.
"""

import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm

from utils import print_or_log, setup_logger, timing_decorator

DEFAULT_BASELINE_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/PubMed_FTP_20260503/baseline_parsed_dfs"
)
DEFAULT_UPDATEFILES_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/PubMed_FTP_20260503/updatefiles_parsed_dfs"
)
DEFAULT_OUTPUT_TSV = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/PubMed_FTP_20260503/"
    "integrated_structured_abstracts.tsv"
)


def _list_tsv_paths(directory):
    pattern = os.path.join(directory, "*.tsv")
    return sorted(glob.glob(pattern))


def _read_tsv_normalized(path):
    df = pd.read_csv(path, sep="\t")
    if "pmid" in df.columns:
        df["pmid"] = df["pmid"].astype(str)
    return df


@timing_decorator
def merge_baseline_and_updates(
    baseline_dir,
    updatefiles_dir,
    output_tsv,
    logger=None,
):
    baseline_paths = _list_tsv_paths(baseline_dir)
    update_paths = _list_tsv_paths(updatefiles_dir)
    print_or_log(
        logger,
        f"* baseline TSVs: {len(baseline_paths)} under {baseline_dir}",
    )
    print_or_log(
        logger,
        f"* updatefile TSVs: {len(update_paths)} under {updatefiles_dir}",
    )
    if not baseline_paths:
        raise FileNotFoundError(f"No *.tsv files under {baseline_dir}")

    parts = []
    for path in tqdm(baseline_paths, desc="baseline chunks"):
        parts.append(_read_tsv_normalized(path))
    integrated = pd.concat(parts, ignore_index=True)
    integrated = integrated.drop_duplicates(subset=["pmid"], keep="last")
    print_or_log(
        logger,
        f"* integrated baseline rows (unique pmid): {len(integrated)}",
    )

    for path in tqdm(update_paths, desc="update chunks"):
        chunk = _read_tsv_normalized(path)
        integrated = pd.concat([integrated, chunk], ignore_index=True)
        integrated = integrated.drop_duplicates(subset=["pmid"], keep="last")

    out_dir = os.path.dirname(output_tsv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    integrated.to_csv(output_tsv, sep="\t", index=False)
    print_or_log(
        logger,
        f"* wrote {len(integrated)} rows to {output_tsv}",
    )
    return integrated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge baseline parsed TSVs with updatefile TSVs in filename order "
            "(last row per PMID wins)."
        )
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing baseline *.tsv chunk files",
    )
    parser.add_argument(
        "--updatefiles_dir",
        type=str,
        default=DEFAULT_UPDATEFILES_DIR,
        help="Directory containing updatefile *.tsv chunk files",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default=DEFAULT_OUTPUT_TSV,
        help="Path for the single merged TSV output",
    )
    parser.add_argument(
        "--log_file_path",
        "-log",
        type=str,
        default=None,
        help="Optional log file path (uses setup_logger from utils if set)",
    )
    args = parser.parse_args()

    logger = None
    if args.log_file_path:
        logger = setup_logger(args.log_file_path)

    merge_baseline_and_updates(
        args.baseline_dir,
        args.updatefiles_dir,
        args.output_tsv,
        logger=logger,
    )


if __name__ == "__main__":
    main()
