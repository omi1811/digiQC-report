#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import pandas as pd


OPEN_STATUSES = {"RAISED", "REJECTED", "RESPONDED"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate open-issues mapping by Building (Location L1) and Assigned Team "
            "for Current Status in Raised/Rejected/Responded."
        )
    )
    p.add_argument(
        "--input",
        "-i",
        default="",
        help="Path to Issues/Instructions CSV. If omitted, auto-detects CSV-INSTRUCTION*.csv in current directory.",
    )
    p.add_argument(
        "--output-prefix",
        "-o",
        default="",
        help="Optional output prefix. If omitted, derived from Location L0 (or 'issues').",
    )
    return p.parse_args()


def read_csv_robust(path: str) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for sep in (",", "\t", None):
        try:
            if sep is None:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
            return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine="python")
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError(f"Could not read file: {path}")


def autodetect_input() -> str:
    patterns = [
        "CSV-INSTRUCTION*.csv",
        "*Instruction*.csv",
        "*INSTRUCTION*.csv",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No instruction/issues CSV found. Pass --input explicitly (example: CSV-INSTRUCTION-....csv)."
        )
    return candidates[0]


def normalize_text_col(df: pd.DataFrame, col: str, empty_label: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([empty_label] * len(df), index=df.index, dtype=str)
    s = df[col].astype(str).str.strip()
    return s.replace("", empty_label)


def main() -> None:
    args = parse_args()
    in_path = args.input.strip() or autodetect_input()
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = read_csv_robust(in_path)

    required = ["Current Status", "Location L1", "Assigned Team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    status = df["Current Status"].astype(str).str.strip().str.upper()
    open_df = df[status.isin(OPEN_STATUSES)].copy()

    open_df["Building"] = normalize_text_col(open_df, "Location L1", "UNKNOWN")
    open_df["Assigned Team"] = normalize_text_col(open_df, "Assigned Team", "UNASSIGNED")
    open_df["Current Status"] = open_df["Current Status"].astype(str).str.strip().str.upper()

    by_bldg_team_status = (
        open_df.groupby(["Building", "Assigned Team", "Current Status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["RAISED", "REJECTED", "RESPONDED"]:
        if col not in by_bldg_team_status.columns:
            by_bldg_team_status[col] = 0

    by_bldg_team_status["Open Total"] = (
        by_bldg_team_status["RAISED"].astype(int) + by_bldg_team_status["REJECTED"].astype(int)
        + by_bldg_team_status["RESPONDED"].astype(int)
    )

    by_bldg_team_status = by_bldg_team_status[
        ["Building", "Assigned Team", "RAISED", "REJECTED", "RESPONDED", "Open Total"]
    ].sort_values(["Building", "Open Total", "Assigned Team"], ascending=[True, False, True])

    matrix = (
        by_bldg_team_status.pivot_table(
            index="Building",
            columns="Assigned Team",
            values="Open Total",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    if args.output_prefix.strip():
        prefix = args.output_prefix.strip()
    else:
        if "Location L0" in df.columns:
            loc0 = df["Location L0"].astype(str).str.strip()
            names = [x for x in loc0.unique().tolist() if x]
            prefix = names[0].replace(" ", "_") if names else "issues"
        else:
            prefix = "issues"

    map_out = f"{prefix}-open-issues-mapping-by-building-team.csv"
    matrix_out = f"{prefix}-open-issues-matrix-building-vs-assigned-team.csv"

    by_bldg_team_status.to_csv(map_out, index=False)
    matrix.reset_index().to_csv(matrix_out, index=False)

    print(f"Input file: {in_path}")
    print(f"Open issues considered (Raised/Rejected): {len(open_df)}")
    print(f"Wrote mapping: {map_out}")
    print(f"Wrote matrix : {matrix_out}")


if __name__ == "__main__":
    main()
