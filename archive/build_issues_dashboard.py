#!/usr/bin/env python3
"""
Issues-only dashboard (Quality, Safety excluded)

Reads Combined_Instructions.csv and prints per-project counts for Today, This Month, and Cumulative,
separately for External and Internal categories.

This was extracted from build_dashboard.py to keep that script EQC-only.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Dict, List
import io
import os
import re

import pandas as pd

import analysis_issues as ISS


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issues-only dashboard from Combined_Instructions.csv")
    p.add_argument("--issues", default="Combined_Instructions.csv", help="Path to combined Instructions file")
    p.add_argument("--date", "-d", help="Override today's date (dd-mm-yyyy)")
    p.add_argument("--projects", nargs="*", help="Optional list of project names to include (match against Location L0)")
    p.add_argument("--external-name", default="Omkar", help="External hard match name for Instructions")
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


def _projects_from(df: pd.DataFrame) -> List[str]:
    vals = df.get("Location L0", df.get("Project Name", "")).astype(str).str.strip()
    return [p for p in sorted(vals.unique()) if p]


def _read_instructions_robust(path: str) -> pd.DataFrame:
    df = ISS._read_instructions(path)
    if df.shape[1] > 1:
        return df
    # Attempt header-merge fallback
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        m = re.search(r"(?m)^\s*\d{2,},", text)
        if m:
            pos = m.start()
            header = text[:pos]
            data = text[pos:]
            header = header.replace("\r", "").replace("\n", "")
            new_text = header + "\n" + data
            buf = io.StringIO(new_text)
            df2 = pd.read_csv(buf, dtype=str, keep_default_na=False, engine='python')
            if df2.shape[1] > 1:
                return df2
    except Exception:
        pass
    return df


essueSummary = Dict[str, Dict[str, Dict[str, ISS.Counts]]]


def issues_summary_by_project(path: str, target: date, external_name: str, projects_filter: List[str] | None) -> essueSummary:
    df = _read_instructions_robust(path)
    df = _drop_demo_rows(df)
    df = ISS._filter_quality(df)
    df = ISS._tag_external(df, external_name)

    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)

    all_projects = _projects_from(df)
    if projects_filter:
        wanted = set(projects_filter)
        projects = [p for p in all_projects if p in wanted]
    else:
        projects = all_projects

    out: essueSummary = {}
    for proj in projects:
        if "Location L0" in df.columns:
            proj_mask = df["Location L0"].astype(str).str.strip().eq(proj)
        elif "Project Name" in df.columns:
            proj_mask = df["Project Name"].astype(str).str.strip().eq(proj)
        else:
            continue
        sub = df[proj_mask]
        masks = ISS._timeframe_masks(dates, target)
        cat_out: Dict[str, Dict[str, ISS.Counts]] = {}
        for category, cat_df in sub.groupby("__Category"):
            res: Dict[str, ISS.Counts] = {}
            for tf, mask in masks.items():
                mask_aligned = mask.reindex(cat_df.index, fill_value=False)
                res[tf] = ISS._count_frame(cat_df[mask_aligned])
            cat_out[category] = res
        for cat in ("External", "Internal"):
            if cat not in cat_out:
                cat_out[cat] = {tf: ISS.Counts(0, 0, 0) for tf in ("today", "month", "all")}
        out[proj] = cat_out
    return out


def print_issues_console(issues: essueSummary, target: date) -> None:
    for proj in sorted(issues.keys()):
        print(f"Project: {proj}")
        print("  Issues (Quality, Safety excluded):")
        for cat in ("External", "Internal"):
            r = issues[proj][cat]
            t_today = r["today"]; t_mon = r["month"]; t_all = r["all"]
            print(f"    {cat}:")
            print(f"      Today → Total: {t_today.total}, Open: {t_today.open}, Closed: {t_today.closed}")
            print(f"      This Month → Total: {t_mon.total}, Open: {t_mon.open}, Closed: {t_mon.closed}")
            print(f"      Cumulative → Total: {t_all.total}, Open: {t_all.open}, Closed: {t_all.closed}")
        print()


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    target = _today(args.date)
    issues_path = args.issues
    if not os.path.exists(issues_path):
        pass
    issues_res = issues_summary_by_project(issues_path, target, args.external_name, args.projects)
    print_issues_console(issues_res, target)


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
