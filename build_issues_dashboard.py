#!/usr/bin/env python3
"""
Issues-only console dashboard (Quality issues only; Safety excluded)

Reads Combined_Instructions.csv and prints per-project counts aggregated across ALL issues
for Today, This Month, and Cumulative, mirroring the style and robustness used in
`build_dashboard.py` (canonical project mapping, DEMO filter, date override, and path fallback).

This is split from the EQC dashboard so each domain can evolve independently.
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
try:
    # Prefer shared function from repo root when available
    from project_utils import canonical_project_from_row  # type: ignore
except Exception:
    # Fallback: derive a canonical project from typical columns without repo utils
    def canonical_project_from_row(row: pd.Series) -> str:  # type: ignore
        for c in ("Location L0", "Project Name", "Project"):
            if c in row and str(row[c]).strip():
                return str(row[c]).strip()
        # As a last resort, attempt to extract leading token from Location / Reference
        val = str(row.get("Location / Reference", "")).strip()
        if "/" in val:
            return val.split("/", 1)[0].strip()
        return val or ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issues-only dashboard from Combined_Instructions.csv")
    p.add_argument("--issues", default="Combined_Instructions.csv", help="Path to combined Instructions file")
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
    # Use a canonical key (falls back to Project when needed) like build_dashboard.py
    keys = df.apply(_canonical_project_from_row, axis=1)
    vals = keys.astype(str).str.strip()
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


IssueSummary = Dict[str, Dict[str, ISS.Counts]]


def issues_summary_by_project(path: str, target: date, projects_filter: List[str] | None) -> IssueSummary:
    df = _read_instructions_robust(path)
    df = _drop_demo_rows(df)
    df = ISS._filter_quality(df)

    # Parse dates once for timeframe masks
    dates = (
        df.get("Raised On Date").map(ISS._parse_date_safe)
        if "Raised On Date" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    # Build canonical project keys
    df = df.copy()
    df["__ProjectKey"] = df.apply(_canonical_project_from_row, axis=1)

    all_projects = _projects_from(df)
    if projects_filter:
        wanted = set(projects_filter)
        projects = [p for p in all_projects if p in wanted]
    else:
        projects = all_projects

    out: IssueSummary = {}
    for proj in projects:
        proj_mask = df["__ProjectKey"].astype(str).str.strip().eq(proj)
        sub = df[proj_mask]
        masks = ISS._timeframe_masks(dates, target)
        res: Dict[str, ISS.Counts] = {}
        for tf, mask in masks.items():
            mask_aligned = mask.reindex(sub.index, fill_value=False)
            res[tf] = ISS._count_frame(sub[mask_aligned])
        out[proj] = res
    return out


def print_issues_console(issues: IssueSummary, target: date) -> None:
    for proj in sorted(issues.keys()):
        print(f"Project: {proj}")
        print("  Issues (Quality only; Safety excluded):")
        r = issues[proj]
        t_today = r["today"]; t_mon = r["month"]; t_all = r["all"]
        print(f"    Today ({target.strftime('%d-%m-%Y')}) → Total: {t_today.total}, Open: {t_today.open}, Closed: {t_today.closed}")
        print(f"    This Month → Total: {t_mon.total}, Open: {t_mon.open}, Closed: {t_mon.closed}")
        print(f"    Cumulative → Total: {t_all.total}, Open: {t_all.open}, Closed: {t_all.closed}")
        print()


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    target = _today(args.date)
    issues_path = args.issues
    # Auto-detect common typos/alternates for Combined_Instructions.* if not found
    if not os.path.exists(issues_path):
        alt = os.path.splitext(issues_path)[0] + ".scv"
        if os.path.exists(alt):
            issues_path = alt
    # If still not found, try to autodiscover a likely Instructions CSV (latest by mtime)
    if not os.path.exists(issues_path):
        candidates: list[str] = []
        try:
            for root, _dirs, files in os.walk(os.getcwd()):
                for f in files:
                    fl = f.lower()
                    if not fl.endswith(".csv"):
                        continue
                    if fl.startswith("csv-instruction-latest-report") or ("instruction" in fl):
                        candidates.append(os.path.join(root, f))
        except Exception:
            candidates = []
        if candidates:
            # pick the most recently modified file
            issues_path = max(candidates, key=lambda p: os.path.getmtime(p))
            print(f"[info] Using autodetected issues file: {issues_path}")
        else:
            print("[warn] No issues CSV found. Provide --issues path to a Combined_Instructions.csv.")
            print()
            return
    issues_res = issues_summary_by_project(issues_path, target, args.projects)
    if not issues_res:
        print(f"[info] No issues to report for {target.strftime('%d-%m-%Y')} (file empty or all filtered as Safety)")
        return
    print_issues_console(issues_res, target)


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
