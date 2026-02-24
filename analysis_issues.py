#!/usr/bin/env python3
"""
Issues analysis (Quality dashboard)

Reads instruction CSV/TSV extracts (e.g., CityLifeInstruction.csv, FuturaInstruction.csv,
PalacioInstruction.csv) and prints counts for:
- Total issues
- Open (Current Status in RAISED, REJECTED)
- Closed (Current Status in CLOSED, RESPONDED)

Scoping and rules:
- Exclude Safety: drop rows where Type L0 contains the word "Safety" (case-insensitive)
- External vs Internal: External if Raised By contains the name 'Omkar' (case-insensitive substring);
  all others are Internal. (You can change the external name with --external-name)
- Time frames based on Raised On Date:
  * Today (== target date; default = system today; override with --date dd-mm-yyyy)
  * This Month (same calendar month & year as target date)
  * Cumulative (all rows)

Output is plain stdout, similar to analysis_eqc.py. No changes to Futura_modified.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from datetime import date, datetime
from typing import Dict, List

import pandas as pd

OPEN_STATUSES = {"RAISED", "REJECTED"}
CLOSED_STATUSES = {"CLOSED", "RESPONDED"}


@dataclass
class Counts:
    total: int
    open: int
    closed: int


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issues analysis (Quality dashboard)")
    p.add_argument("files", nargs="+", help="Instruction CSV/TSV files (CityLifeInstruction.csv, FuturaInstruction.csv, PalacioInstruction.csv)")
    p.add_argument("--date", "-d", help="Override today's date (dd-mm-yyyy)")
    p.add_argument("--external-name", default="Omkar", help="Name treated as External in 'Raised By' (hard match, case-insensitive)")
    return p.parse_args(argv)


def _read_instructions(path: str) -> pd.DataFrame:
    # If the file is empty, return an empty DataFrame
    try:
        if os.path.exists(path) and os.path.getsize(path) == 0:
            return pd.DataFrame()
    except Exception:
        pass
    last_err: Exception | None = None
    for sep, engine in [("\t", "python"), (",", "python"), (None, "python")]:
        try:
            if sep is None:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine=engine)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine=engine)
            return df
        except Exception as e:
            last_err = e
    # If we still failed, treat unreadable as empty to keep pipeline robust
    return pd.DataFrame()


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


def _filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude Safety and DEMO from quality issues.
    
    Filter order (matches manual Excel filtering):
    1. Remove all rows where Type L0 contains 'safety' (519 rows)
    2. Remove DEMO from 'Project Name' column (101 rows after safety removal)
    
    Result: 2819 total → 2300 (after safety) → 2199 (after DEMO) = Quality issues only
    """
    # Step 1: Exclude all Safety-related issues first
    if "Type L0" in df.columns:
        mask_safety = df["Type L0"].astype(str).str.contains('safety', case=False, na=False)
        df = df[~mask_safety]
    
    # Step 2: Then exclude DEMO projects from Project Name
    if "Project Name" in df.columns:
        mask_demo = df["Project Name"].astype(str).str.contains('DEMO', case=False, na=False)
        df = df[~mask_demo]
    
    return df


def _tag_external(df: pd.DataFrame, external_name: str) -> pd.DataFrame:
    """Tag issues as External vs Internal exactly (no partials).

    - Blank external_name => all Internal
    - Case-insensitive exact match on 'Raised By' after stripping
    - Multiple names supported via comma or pipe separators
    """
    target_raw = (external_name or "").strip()
    if "Raised By" not in df.columns:
        df["Raised By"] = ""
    if target_raw == "":
        df["__External"] = False
        df["__Category"] = "Internal"
        return df
    import re as _re
    tokens = [t.strip() for t in _re.split(r"[\|,]", target_raw) if t.strip()]
    up_tokens = [t.upper() for t in tokens]

    def _is_external(name: str) -> bool:
        s = (name or "").strip().upper()
        for tok in up_tokens:
            if tok and s == tok:
                return True
        return False

    df["__External"] = df["Raised By"].astype(str).map(_is_external)
    df["__Category"] = df["__External"].map(lambda x: "External" if x else "Internal")
    return df



def _status_bucket(s: str) -> str:
    su = (s or "").strip().upper()
    if su in OPEN_STATUSES:
        return "Open"
    if su in CLOSED_STATUSES:
        return "Closed"
    return "Other"


UPDATED_DATE_COL = "Current Status Updated On (Date)"


def _count_frame(
    df: pd.DataFrame,
    parent_df: "pd.DataFrame | None" = None,
    close_mask: "pd.Series | None" = None,
) -> Counts:
    """Count raised/open/closed observations.

    Args:
        df          : Rows raised in the time-period (filtered by Raised On Date).
        parent_df   : Full group (project+category) used to count closed-in-period.
                      When provided together with close_mask, closed count =
                      rows in parent_df whose Updated Date matches the period.
        close_mask  : Boolean Series aligned to parent_df.index — True = Updated
                      Date falls in the period.  Ignored when parent_df is None.
    """
    total = len(df)
    if "Current Status" not in df.columns:
        return Counts(total=total, open=0, closed=0)
    buckets = df["Current Status"].map(_status_bucket)
    open_cnt = int((buckets == "Open").sum())
    if close_mask is not None and parent_df is not None and "Current Status" in parent_df.columns:
        # Closed count: items closed IN the period (regardless of when they were raised)
        parent_buckets = parent_df["Current Status"].map(_status_bucket)
        close_mask_aligned = close_mask.reindex(parent_df.index, fill_value=False)
        closed_cnt = int(((parent_buckets == "Closed") & close_mask_aligned).sum())
    else:
        closed_cnt = int((buckets == "Closed").sum())
    return Counts(total=total, open=open_cnt, closed=closed_cnt)


def _timeframe_masks(dates: pd.Series, target: date) -> Dict[str, pd.Series]:
    today_mask = dates == target
    month_mask = dates.map(lambda d: (d is not None) and (d.year == target.year and d.month == target.month))
    all_mask = pd.Series([True] * len(dates), index=dates.index)
    return {"today": today_mask, "month": month_mask, "all": all_mask}


def analyze_file(path: str, target: date, external_name: str) -> Dict[str, Dict[str, Counts]]:
    """Return nested dict: {category: {timeframe: Counts}}"""
    df = _read_instructions(path)
    df = _filter_quality(df)
    df = _tag_external(df, external_name)

    # Parse dates
    if "Raised On Date" in df.columns:
        dates = df["Raised On Date"].map(_parse_date_safe)
    else:
        dates = pd.Series([None] * len(df))

    masks = _timeframe_masks(dates, target)
    # Build close masks using "Current Status Updated On (Date)" for closed-in-period counts
    if UPDATED_DATE_COL in df.columns:
        close_dates = df[UPDATED_DATE_COL].map(_parse_date_safe)
    else:
        close_dates = pd.Series([None] * len(df), index=df.index)
    close_masks = _timeframe_masks(close_dates, target)
    out: Dict[str, Dict[str, Counts]] = {}
    for category, cat_df in df.groupby("__Category"):
        cat_res: Dict[str, Counts] = {}
        for tf, mask in masks.items():
            # align mask to the subset index to avoid reindex warnings
            mask_aligned = mask.reindex(cat_df.index, fill_value=False)
            sub = cat_df[mask_aligned]
            if tf != "all":
                close_mask_aligned = close_masks[tf].reindex(cat_df.index, fill_value=False)
                cat_res[tf] = _count_frame(sub, parent_df=cat_df, close_mask=close_mask_aligned)
            else:
                cat_res[tf] = _count_frame(sub)
        out[category] = cat_res
    # Ensure both categories present even if empty
    for cat in ("External", "Internal"):
        if cat not in out:
            out[cat] = {tf: Counts(0, 0, 0) for tf in ("today", "month", "all")}
    return out


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    # Determine target date
    if args.date:
        try:
            target = datetime.strptime(args.date, "%d-%m-%Y").date()
        except Exception:
            target = date.today()
    else:
        target = date.today()

    for f in args.files:
        try:
            res = analyze_file(f, target, args.external_name)
            print(f"File: {f}")
            for cat in ("External", "Internal"):
                r = res[cat]
                t_today = r["today"]; t_mon = r["month"]; t_all = r["all"]
                print(f"  {cat}:")
                print(f"    Today's Observation ({target.strftime('%d-%m-%Y')}): Total: {t_today.total}, Open: {t_today.open}, Closed: {t_today.closed}")
                print(f"    This Month Observation: Total: {t_mon.total}, Open: {t_mon.open}, Closed: {t_mon.closed}")
                print(f"    Cumulative: Total: {t_all.total}, Open: {t_all.open}, Closed: {t_all.closed}")
            print()
        except Exception as e:
            print(f"File: {f}")
            print("  Error:", str(e))
            print()


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
