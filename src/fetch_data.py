"""Download clinical trial records from the ClinicalTrials.gov API (v2).

The script fetches a configurable number of studies, keeps only the fields we
need for the project, and stores them locally as CSV. Raw data is intentionally
kept out of version control (see .gitignore); anyone can reproduce the dataset
by running this script.

Example
-------
    python -m src.fetch_data --limit 2000 --output data/raw/trials.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Only studies with a final outcome are useful for the Completed-vs-Terminated
# target. Intermediate statuses (recruiting, active, etc.) are excluded.
FINAL_STATUSES = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"]

# Columns written to the CSV. `why_stopped` is collected for analysis only and
# must NOT be used as a model feature (data leakage). See PROJECT-PLAN.md.
CSV_FIELDS = [
    "nct_id",
    "overall_status",
    "why_stopped",
    "brief_summary",
    "detailed_description",
    "eligibility_criteria",
    "study_type",
    "phase",
    "enrollment_count",
    "conditions",
    "intervention_types",
]


def _get(section: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Safely walk nested dictionaries, returning `default` if any key is missing."""
    node: Any = section
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default if key == keys[-1] else {})
    return node if node is not None else default


def parse_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten one API study record into the columns we care about."""
    protocol = study.get("protocolSection", {})

    phases = _get(protocol, "designModule", "phases", default=[])
    conditions = _get(protocol, "conditionsModule", "conditions", default=[])
    interventions = _get(protocol, "armsInterventionsModule", "interventions", default=[])
    intervention_types = [i.get("type", "") for i in interventions if isinstance(i, dict)]

    return {
        "nct_id": _get(protocol, "identificationModule", "nctId"),
        "overall_status": _get(protocol, "statusModule", "overallStatus"),
        "why_stopped": _get(protocol, "statusModule", "whyStopped"),
        "brief_summary": _get(protocol, "descriptionModule", "briefSummary"),
        "detailed_description": _get(protocol, "descriptionModule", "detailedDescription"),
        "eligibility_criteria": _get(protocol, "eligibilityModule", "eligibilityCriteria"),
        "study_type": _get(protocol, "designModule", "studyType"),
        "phase": "|".join(phases) if isinstance(phases, list) else "",
        "enrollment_count": _get(protocol, "designModule", "enrollmentInfo", "count"),
        "conditions": "|".join(conditions) if isinstance(conditions, list) else "",
        "intervention_types": "|".join(intervention_types),
    }


def iter_studies(
    limit: int,
    statuses: list[str],
    page_size: int = 1000,
    timeout: int = 30,
) -> Iterator[dict[str, Any]]:
    """Yield parsed studies up to `limit`, handling API pagination."""
    session = requests.Session()
    next_token: str | None = None
    fetched = 0

    while fetched < limit:
        params: dict[str, Any] = {
            "filter.overallStatus": ",".join(statuses),
            "pageSize": min(page_size, limit - fetched),
            "format": "json",
        }
        if next_token:
            params["pageToken"] = next_token

        response = session.get(API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        studies = payload.get("studies", [])
        if not studies:
            break

        for study in studies:
            yield parse_study(study)
            fetched += 1
            if fetched >= limit:
                break

        next_token = payload.get("nextPageToken")
        if not next_token:
            break

        # Be polite to the public API.
        time.sleep(0.2)


def write_csv(rows: Iterator[dict[str, Any]], output: Path) -> int:
    """Write parsed rows to CSV, returning the number written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of studies to download (default: 2000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/trials.csv"),
        help="Output CSV path (default: data/raw/trials.csv).",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=FINAL_STATUSES,
        help=f"Overall statuses to include (default: {FINAL_STATUSES}).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="API page size, max 1000 (default: 1000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Fetching up to {args.limit} studies with statuses {args.statuses} ...")
    rows = iter_studies(limit=args.limit, statuses=args.statuses, page_size=args.page_size)
    written = write_csv(rows, args.output)
    print(f"Wrote {written} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
