import argparse
import json
import os
import sys

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from prompts import required_response_format, system_prompt, user_prompt

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("endpoint"),
    api_key=os.getenv("api_key"),
)


def extract_structured_abstracts_via_GPT(abstract: str, deployment_name: str) -> str:
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt.format(abstract=abstract),
            },
        ],
        response_format=required_response_format,
    )

    return completion.choices[0].message.content


def _normalize_predicted_json(raw: str | None) -> str:
    """Return a compact JSON object string, or '{}' if invalid."""
    if raw is None:
        return "{}"
    s = str(raw).strip()
    if not s:
        return "{}"
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return "{}"
    if not isinstance(obj, dict):
        return "{}"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def process_tsv(
    input_tsv: str,
    output_tsv: str,
    deployment_name: str,
    limit: int | None = None,
    resume: bool = False,
) -> None:
    df = pd.read_csv(input_tsv, sep="\t", dtype={"pmid": str})
    if "pmid" not in df.columns:
        raise SystemExit(f"Input TSV missing required column 'pmid': {input_tsv}")
    if "abstract" not in df.columns:
        raise SystemExit(f"Input TSV missing required column 'abstract': {input_tsv}")

    df = df.dropna(subset=["abstract"]).copy()
    df["abstract"] = df["abstract"].astype(str).str.strip()
    df = df[df["abstract"] != ""]

    if limit is not None:
        df = df.head(int(limit))

    processed_pmids: set[str] = set()
    if resume and os.path.isfile(output_tsv):
        try:
            existing = pd.read_csv(output_tsv, sep="\t", dtype={"pmid": str})
            if "pmid" in existing.columns:
                processed_pmids = set(existing["pmid"].astype(str))
            else:
                pd.DataFrame(columns=["pmid", "predicted_structured_abstract"]).to_csv(
                    output_tsv, sep="\t", index=False
                )
        except (pd.errors.EmptyDataError, ValueError):
            processed_pmids = set()
            pd.DataFrame(columns=["pmid", "predicted_structured_abstract"]).to_csv(
                output_tsv, sep="\t", index=False
            )
    else:
        if os.path.isfile(output_tsv):
            os.remove(output_tsv)
        pd.DataFrame(columns=["pmid", "predicted_structured_abstract"]).to_csv(
            output_tsv, sep="\t", index=False
        )

    todo = df[~df["pmid"].astype(str).isin(processed_pmids)]
    if todo.empty:
        print("No rows to process (all pmids already in output or no data).", file=sys.stderr)
        return

    for row in tqdm(
        todo.itertuples(index=False),
        total=len(todo),
        desc="extract_structured_abstracts_via_GPT",
    ):
        pmid = str(row.pmid)
        abstract = str(row.abstract)
        try:
            raw = extract_structured_abstracts_via_GPT(abstract, deployment_name)
        except Exception as exc:  # noqa: BLE001 — log and continue
            print(f"pmid={pmid}: API error: {exc}", file=sys.stderr)
            raw = None
        pred_json = _normalize_predicted_json(raw)
        if pred_json == "{}" and raw is not None and str(raw).strip():
            print(
                f"pmid={pmid}: model response was not a valid JSON object; wrote {{}}",
                file=sys.stderr,
            )
        out_row = pd.DataFrame(
            [{"pmid": pmid, "predicted_structured_abstract": pred_json}]
        )
        out_row.to_csv(output_tsv, sep="\t", index=False, header=False, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read a TSV with pmid and abstract, call the model per row, "
            "and write pmid and predicted_structured_abstract (JSON) for eval.py."
        )
    )
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="TSV with columns pmid and abstract.",
    )
    parser.add_argument(
        "--output_tsv",
        required=True,
        help="Output TSV with columns pmid and predicted_structured_abstract.",
    )
    parser.add_argument(
        "--deployment_name",
        required=True,
        help="Model / deployment name passed to the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many rows (after dropping empty abstracts).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip pmids already present and append new rows.",
    )
    args = parser.parse_args()
    process_tsv(
        args.input_tsv,
        args.output_tsv,
        args.deployment_name,
        limit=args.limit,
        resume=args.resume,
    )
