#!/usr/bin/env python3
"""
ALL-only EQC stage summary

Reads one or more EQC CSV files (comma-separated project extracts such as
PalacioEQC.csv, FuturaEQC.csv, CityLifeEQC.csv) and prints overall counts for
Pre, During, and Post, using this mapping:

- Reinforcement → Pre
- Shuttering → Pre
- Stages containing 'pre' (e.g., 'Pre', 'Pre check', 'Pre Plastering') → Pre
- Stages containing 'during' → During
- Stages containing 'post' → Post
- Anything else (e.g., names not containing pre/during/post, and not reinforcement/shuttering)
    is treated as Other and contributes only to the Post column in the final output
    (per: "The other EQC's are only in post stage and not in Pre and During").

No changes are made to Futura_modified.py. Output is plain stdout for validation.
"""

from __future__ import annotations

import argparse
from typing import Dict, List
from datetime import datetime, date

import pandas as pd


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALL-only EQC stage summary")
    p.add_argument("files", nargs="+", help="One or more EQC CSV files (comma-separated)")
    p.add_argument("--date", "-d", help="Override today's date (dd-mm-yyyy); default = system date")
    return p.parse_args(argv)


def normalize_stage(stage: str) -> str:
    s = str(stage or "").strip().lower()
    # Explicit overrides first
    if "reinforce" in s:
        return "Pre"
    if "shutter" in s:
        return "Pre"
    # Standard mappings
    if "pre" in s or "before" in s:
        return "Pre"
    if "during" in s:
        return "During"
    if "post" in s:
        return "Post"
    # Other stays distinct; we will add it only to Post column later
    return "Other"


def _read_and_clean(path: str) -> pd.DataFrame:
    # Robust CSV/TSV reader: try TSV first, then CSV, then auto-detect
    last_err: Exception | None = None
    for sep, engine in [("\t", "python"), (",", "python"), (None, "python")]:
        try:
            if sep is None:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine=engine)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine=engine)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err if last_err else RuntimeError("Failed to read file")
    # Validate schema: Expect EQC-style columns (e.g., 'Stage', 'Eqc Type').
    # If missing, provide a helpful error instead of computing misleading totals.
    expected_any = ["Stage", "Eqc Type"]
    if not any(col in df.columns for col in expected_any):
        sample_cols = ", ".join(list(df.columns)[:10])
        raise ValueError(
            "Input appears not to be an EQC extract (missing 'Stage'/'Eqc Type'). "
            "Did you pass an Instructions file by mistake? First columns: " + sample_cols
        )
    # Drop known footer/summary rows (often have totals at the end)
    for footer_col in ["Total EQC Stages", "Fail Stages", "% Fail"]:
        if footer_col in df.columns:
            # Keep rows where footer_col is empty; drop if it has a value
            df = df[df[footer_col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "-"])]
    # Also drop rows that clearly have no meaningful content (no Stage and no Eqc Type)
    if "Stage" in df.columns and "Eqc Type" in df.columns:
        mask_empty_stage_and_type = (
            df["Stage"].astype(str).str.strip().eq("") &
            df["Eqc Type"].astype(str).str.strip().eq("")
        )
        if mask_empty_stage_and_type.any():
            df = df[~mask_empty_stage_and_type]
    # Normalize stages into one of: Pre, During, Post, Other
    if "Stage" not in df.columns:
        sample_cols = ", ".join(list(df.columns)[:10])
        raise ValueError(
            "EQC file missing 'Stage' column required for Pre/During/Post mapping. "
            "First columns: " + sample_cols
        )
    df["__Stage"] = df["Stage"].map(normalize_stage)
    return df


def _compute_counts_from_frame(df: pd.DataFrame) -> Dict[str, int]:
    c = df.groupby("__Stage", dropna=False)["__Stage"].count()
    n_pre = int(c.get("Pre", 0))
    n_during = int(c.get("During", 0))
    n_post = int(c.get("Post", 0))
    n_other = int(c.get("Other", 0))

    # Cumulative logic (updated):
    # - Pre = Pre + During + Post + Other (Reinforcement/Shuttering are already counted in Pre via normalization)
    # - During = During + Post + Other
    # - Post = Post + Other
    pre_out = n_pre + n_during + n_post + n_other
    during_out = n_during + n_post + n_other
    post_out = n_post + n_other

    return {"Pre": pre_out, "During": during_out, "Post": post_out}


def summarize_all(path: str) -> Dict[str, int]:
    df = _read_and_clean(path)
    return _compute_counts_from_frame(df)


def _parse_date_safe(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def summarize_today(path: str, target: date) -> Dict[str, int] | Dict:
    df = _read_and_clean(path)
    # Parse Date column and filter
    if "Date" not in df.columns:
        # No date column; nothing is counted as today
        empty = pd.DataFrame(columns=df.columns)
        return {"Pre": 0, "During": 0, "Post": 0, "__total_today_rows": 0}
    dates = df["Date"].astype(str).map(_parse_date_safe)
    df_today = df[dates == target]
    counts = _compute_counts_from_frame(df_today)
    counts["__total_today_rows"] = int(len(df_today))
    return counts


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    # Determine target date (today or override)
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%d-%m-%Y").date()
        except Exception:
            target_date = date.today()
    else:
        target_date = date.today()
    for f in args.files:
        try:
            res = summarize_all(f)
            today_res = summarize_today(f, target_date)
            print(f"File: {f}")
            print(f"  Pre: {res['Pre']}, During: {res['During']}, Post: {res['Post']}")
            print(
                f"  Today ({target_date.strftime('%d-%m-%Y')}): "
                f"Pre: {today_res['Pre']}, During: {today_res['During']}, Post: {today_res['Post']}"
            )
            print()
        except Exception as e:
            print(f"File: {f}")
            print("  Error:", str(e))
            print()


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
