#!/usr/bin/env python3
"""
Build combined console dashboard from single combined files:
- Combined_EQC.csv (all projects EQC)
- Combined_Instructions.csv (all projects Instructions)

Rules:
- Exclude DEMO projects: drop rows where Project/Project Name/Location L0 contains 'DEMO'
- EQC: reuse analysis_eqc stage mapping and cumulative logic; report Pre/During/Post for All-time and Today per project
- Issues (Quality): reuse analysis_issues quality logic (Safety excluded) and hard External match ('Omkar');
  report Total/Open/Closed for Today, This Month, Cumulative per project and per category (External, Internal)

Output: plain stdout, grouped by project.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import os
from typing import Dict, List
import io
import re

import pandas as pd

import analysis_eqc as EQC
from project_utils import canonical_project_from_row
# Issues logic removed to keep this script EQC-only.


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined dashboard from Combined_EQC.csv and Combined_Instructions.csv")
    p.add_argument("--eqc", default="Combined_EQC.csv", help="Path to combined EQC file")
    p.add_argument("--date", "-d", help="Override today's date (dd-mm-yyyy)")
    p.add_argument("--projects", nargs="*", help="Optional list of project names to include (match against Location L0)")
    return p.parse_args(argv)


def _today(args_date: str | None) -> date:
    if args_date:
        try:
            return datetime.strptime(args_date, "%d-%m-%Y").date()
        except Exception:
            pass
    return date.today()


def _drop_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Project", "Project Name", "Location L0") if c in df.columns]
    if not cols:
        return df
    mask_demo = pd.Series(False, index=df.index)
    for c in cols:
        mask_demo = mask_demo | df[c].astype(str).str.contains("DEMO", case=False, na=False)
    return df[~mask_demo]


def _canonical_project_from_row(row: pd.Series) -> str:
    # Use shared canonicalization to avoid split projects on the dashboard
    return canonical_project_from_row(row)


def _projects_from(df: pd.DataFrame) -> List[str]:
    # Build a canonical project key that falls back to Project when Location L0 is blank
    keys = df.apply(_canonical_project_from_row, axis=1)
    vals = keys.astype(str).str.strip()
    return [p for p in sorted(vals.unique()) if p]


def eqc_summary_by_project(path: str, target: date, projects_filter: List[str] | None) -> Dict[str, Dict[str, Dict[str, int]]]:
    # Read and clean using analysis_eqc helper, then augment with parsed dates
    df = EQC._read_and_clean(path)
    df = _drop_demo_rows(df)
    # Normalize stage (already done in _read_and_clean), but ensure column exists
    if "__Stage" not in df.columns:
        df["__Stage"] = df.get("Stage", "").map(EQC.normalize_stage)
    # Parse dates for today split
    dates = df.get("Date").astype(str).map(EQC._parse_date_safe) if "Date" in df.columns else pd.Series([None] * len(df), index=df.index)

    # Determine projects using canonical keys
    df = df.copy()
    df["__ProjectKey"] = df.apply(_canonical_project_from_row, axis=1)
    all_projects = _projects_from(df)
    if projects_filter:
        wanted = set(projects_filter)
        projects = [p for p in all_projects if p in wanted]
    else:
        projects = all_projects

    # Daily-only computation override:
    # For the daily EQC report we want: Pre = Pre, During = During, Post = Post + Other
    def _compute_counts_daily(sub_df: pd.DataFrame) -> Dict[str, int]:
        if sub_df is None or sub_df.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        if "__Stage" not in sub_df.columns:
            stages = sub_df.get("Stage", "").map(EQC.normalize_stage)
        else:
            stages = sub_df["__Stage"].astype(str)
        vc = stages.value_counts()
        n_pre = int(vc.get("Pre", 0))
        n_during = int(vc.get("During", 0))
        n_post = int(vc.get("Post", 0))
        n_other = int(vc.get("Other", 0))
        return {"Pre": n_pre, "During": n_during, "Post": n_post + n_other}

    def _compute_counts_raw(sub_df: pd.DataFrame) -> Dict[str, int]:
        """Raw counts using normalized stages (kept for potential future use)."""
        if sub_df is None or sub_df.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        stages = sub_df["__Stage"].astype(str) if "__Stage" in sub_df.columns else sub_df.get("Stage", "").map(EQC.normalize_stage)
        vc = stages.value_counts()
        return {"Pre": int(vc.get("Pre", 0)), "During": int(vc.get("During", 0)), "Post": int(vc.get("Post", 0))}

    def _compute_counts_raw_pdp(sub_df: pd.DataFrame) -> Dict[str, int]:
        """Raw Pre/During/Post counts from original Stage strings only.

        Important: Do NOT treat Reinforcement/Shuttering as Pre here. They should
        fall into 'Other' so the complement method matches workbook cumulative ALL sums.
        """
        if sub_df is None or sub_df.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        s = sub_df.get("Stage", "").astype(str).str.lower()
        pre = s.str.contains("pre", na=False) | s.str.contains("before", na=False)
        # Exclude reinforcement/shuttering from Pre
        pre = pre & ~s.str.contains("reinforce|shutter", na=False)
        during = s.str.contains("during", na=False)
        post = s.str.contains("post", na=False)
        return {"Pre": int(pre.sum()), "During": int(during.sum()), "Post": int(post.sum())}

    def _compute_counts_cumulative_equiv(sub_df: pd.DataFrame) -> Dict[str, int]:
        """Cumulative roll-up via complement to match workbook ALL sums exactly.

        Pre_out = total_rows
        During_out = total_rows - Pre_raw
        Post_out = total_rows - (Pre_raw + During_raw)
        """
        if sub_df is None or sub_df.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        raw = _compute_counts_raw(sub_df)
        total = int(len(sub_df))
        pre_raw = raw.get("Pre", 0)
        during_raw = raw.get("During", 0)
        return {
            "Pre": total,
            "During": total - pre_raw,
            "Post": total - (pre_raw + during_raw),
        }


    def _compute_counts_cumulative_weeklylogic(sub_df: pd.DataFrame) -> Dict[str, int]:
        """Cumulative roll-up using the same stage mapping as Weekly_report.py.

        Mapping:
        - 'pre' -> Pre
        - 'during' -> During
        - 'post' -> Post
        - 'reinforce' -> Reinforcement
        - 'shutter' -> Shuttering
        - else -> Other

        Output:
        - Pre = total rows
        - During = During + Post + Other
        - Post = Post + Other
        """
        if sub_df is None or sub_df.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        s = sub_df.get("Stage", "").astype(str).str.lower()
        pre = s.str.contains("pre", na=False)
        during = s.str.contains("during", na=False)
        post = s.str.contains("post", na=False)
        reinf = s.str.contains("reinforce", na=False)
        shut = s.str.contains("shutter", na=False)
        other = ~(pre | during | post | reinf | shut)
        total = int(len(s))
        d_count = int(during.sum())
        p_count = int(post.sum())
        o_count = int(other.sum())
        return {
            "Pre": total,
            "During": d_count + p_count + o_count,
            "Post": p_count + o_count,
        }

    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for proj in projects:
        proj_mask = df["__ProjectKey"].astype(str).str.strip().eq(proj)
        sub = df[proj_mask]
        sub_today = sub[dates.loc[sub.index] == target]
        # Month mask (same year & month as target)
        dates_sub = dates.loc[sub.index]
        month_mask = dates_sub.apply(lambda d: bool(d and d.year == target.year and d.month == target.month))
        sub_month = sub[month_mask]
        # Total (all-time): cumulative roll-up to match workbook 'Cumulative' ALL sums (weekly report mapping)
        all_counts = _compute_counts_cumulative_weeklylogic(sub)
        # Daily: raw counts with Post += Other
        today_counts = _compute_counts_daily(sub_today)
        # Monthly: raw counts with Post += Other (no cumulative roll-up)
        month_counts = _compute_counts_daily(sub_month)
        out[proj] = {
            "all": all_counts,
            "month": month_counts,
            "today": today_counts,
            "__rows": {"all": int(len(sub)), "month": int(len(sub_month)), "today": int(len(sub_today))},
        }
    return out


def print_eqc_console(eqc: Dict[str, Dict[str, Dict[str, int]]], target: date) -> None:
    projects = sorted(set(eqc.keys()))
    for proj in projects:
        print(f"Project: {proj}")
        if proj in eqc:
            e_all = eqc[proj]["all"]; e_day = eqc[proj]["today"]; e_mon = eqc[proj].get("month", {"Pre": 0, "During": 0, "Post": 0}); e_rows = eqc[proj].get("__rows", {"all": None, "month": None, "today": None})
            print("  EQC:")
            if e_rows.get("all") is not None:
                print(f"    Rows (after DEMO filter) → Total: {e_rows.get('all')}, This Month: {e_rows.get('month')}, Today: {e_rows.get('today')}")
            print(f"    Total → Pre: {e_all['Pre']}, During: {e_all['During']}, Post: {e_all['Post']}")
            print(f"    This Month → Pre: {e_mon['Pre']}, During: {e_mon['During']}, Post: {e_mon['Post']}")
            print(f"    Today ({target.strftime('%d-%m-%Y')}) → Pre: {e_day['Pre']}, During: {e_day['During']}, Post: {e_day['Post']}")
        else:
            print("  EQC: (no data)")
        print()


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    target = _today(args.date)

    eqc_path = args.eqc
    # Auto-detect common typos/alternates for Combined_EQC.* if not found
    if not os.path.exists(eqc_path):
        # Try .scv typo
        alt = os.path.splitext(eqc_path)[0] + ".scv"
        if os.path.exists(alt):
            eqc_path = alt
    try:
        eqc_res = eqc_summary_by_project(eqc_path, target, args.projects)
    except ValueError as ve:
        msg = str(ve)
        raise
    print_eqc_console(eqc_res, target)


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
