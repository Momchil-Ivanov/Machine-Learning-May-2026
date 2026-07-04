"""Clean raw ClinicalTrials.gov records into a model-ready dataset.

Responsibilities
----------------
1. Build the binary label from `overall_status`
   (COMPLETED -> 1; TERMINATED/WITHDRAWN/SUSPENDED -> 0).
2. Merge the free-text fields into a single document.
3. Remove leakage-prone sentences (those that reveal the outcome).
4. Assemble simple tabular features.
5. Pass through `start_date` when present (for temporal external validation).

The leakage removal is controlled by a flag so we can later measure its effect
in an ablation study (model trained with vs. without cleaning).

Example
-------
    python -m src.preprocess --input data/raw/trials.csv \\
        --output data/processed/clean.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Status -> binary label. Statuses not listed here are dropped (no final outcome).
STATUS_TO_LABEL = {
    "COMPLETED": 1,
    "TERMINATED": 0,
    "WITHDRAWN": 0,
    "SUSPENDED": 0,
}

TEXT_FIELDS = ["brief_summary", "detailed_description", "eligibility_criteria"]

# Words whose presence in a sentence strongly hints at the (negative) outcome.
# A sentence containing any of these is dropped when leakage removal is enabled.
LEAKAGE_TERMS = [
    "terminat",   # terminate, terminated, termination
    "withdraw",   # withdraw, withdrawn
    "suspend",    # suspend, suspended
    "stopped",
    "halted",
    "discontinu", # discontinue, discontinued
    "closed",
    "closing",
    "low enrollment",
    "lack of enrollment",
    "lack of funding",
    "slow accrual",
]

# Split text into sentences on ., !, ? boundaries and on newlines.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV produced by fetch_data.py."""
    return pd.read_csv(path)


def make_label(status: str) -> int | None:
    """Map an overall status to a binary label, or None if it has no final outcome."""
    return STATUS_TO_LABEL.get(str(status).strip().upper())


def strip_leakage_sentences(text: str) -> str:
    """Drop sentences that contain any leakage term (case-insensitive)."""
    if not isinstance(text, str) or not text:
        return ""
    sentences = _SENTENCE_SPLIT.split(text)
    kept = [
        sentence
        for sentence in sentences
        if not any(term in sentence.lower() for term in LEAKAGE_TERMS)
    ]
    return " ".join(part.strip() for part in kept if part.strip())


def merge_text(row: pd.Series, remove_leakage: bool) -> str:
    """Concatenate the text fields of one trial into a single document."""
    parts = [str(row.get(field, "") or "") for field in TEXT_FIELDS]
    text = " ".join(part.strip() for part in parts if part.strip())
    if remove_leakage:
        text = strip_leakage_sentences(text)
    return text


def build_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Build simple tabular features from the raw columns."""
    tabular = pd.DataFrame(index=df.index)
    tabular["study_type"] = df["study_type"].fillna("UNKNOWN").astype(str)
    tabular["phase"] = df["phase"].fillna("NA").astype(str)
    tabular["enrollment_count"] = pd.to_numeric(
        df["enrollment_count"], errors="coerce"
    ).fillna(0)

    def _count(value: object) -> int:
        if not isinstance(value, str) or not value:
            return 0
        return len([item for item in value.split("|") if item])

    tabular["n_conditions"] = df["conditions"].apply(_count)
    tabular["n_intervention_types"] = df["intervention_types"].apply(_count)
    return tabular


def preprocess(df: pd.DataFrame, remove_leakage: bool = True) -> pd.DataFrame:
    """Turn raw records into a clean, model-ready DataFrame."""
    result = pd.DataFrame(index=df.index)
    result["nct_id"] = df["nct_id"]
    if "start_date" in df.columns:
        result["start_date"] = df["start_date"].fillna("").astype(str)
    result["label"] = df["overall_status"].apply(make_label)

    # Keep only rows with a usable final outcome.
    result = result[result["label"].notna()].copy()
    result["label"] = result["label"].astype(int)

    df = df.loc[result.index]
    result["text"] = df.apply(lambda row: merge_text(row, remove_leakage), axis=1)
    result = pd.concat([result, build_tabular(df)], axis=1)

    # Drop rows that ended up with no text after cleaning.
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/trials.csv"),
        help="Input CSV path (default: data/raw/trials.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/clean.csv"),
        help="Output CSV path (default: data/processed/clean.csv).",
    )
    parser.add_argument(
        "--keep-leakage",
        action="store_true",
        help="Do NOT remove leakage sentences (for ablation comparison).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    remove_leakage = not args.keep_leakage

    df = load_raw(args.input)
    clean = preprocess(df, remove_leakage=remove_leakage)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)

    pos = int((clean["label"] == 1).sum())
    neg = int((clean["label"] == 0).sum())
    dated = int(clean["start_date"].astype(str).str.len().gt(0).sum()) if "start_date" in clean.columns else 0
    print(
        f"Wrote {len(clean)} rows to {args.output} "
        f"(completed={pos}, not_completed={neg}, remove_leakage={remove_leakage}"
        + (f", with_start_date={dated}" if dated else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
