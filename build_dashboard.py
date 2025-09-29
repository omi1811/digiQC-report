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
import analysis_issues as ISS


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined dashboard from Combined_EQC.csv and Combined_Instructions.csv")
    p.add_argument("--eqc", default="Combined_EQC.csv", help="Path to combined EQC file")
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


def _canonical_project_from_row(row: pd.Series) -> str:
    l0 = str(row.get("Location L0", "") or "").strip()
    if l0:
        return l0
    proj = str(row.get("Project", "") or "").strip()
    # Map common synonyms to canonical keys used elsewhere
    m = {
        "Itrend Futura": "FUTURA",
        "FUTURA": "FUTURA",
        "City Life": "Itrend City Life",
        "Itrend City Life": "Itrend City Life",
        "Itrend Palacio": "Itrend-Palacio",
        "Itrend-Palacio": "Itrend-Palacio",
    }
    return m.get(proj, proj)


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

    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for proj in projects:
        proj_mask = df["__ProjectKey"].astype(str).str.strip().eq(proj)
        sub = df[proj_mask]
        sub_today = sub[dates.loc[sub.index] == target]
        all_counts = EQC._compute_counts_from_frame(sub)
        today_counts = EQC._compute_counts_from_frame(sub_today)
        out[proj] = {
            "all": all_counts,
            "today": today_counts,
            "__rows": {"all": int(len(sub)), "today": int(len(sub_today))},
        }
    return out


def _read_instructions_robust(path: str) -> pd.DataFrame:
    """Read Instructions CSV tolerating a header split across two lines.
    If standard read returns a single-column frame, attempt to merge first two lines of the file
    and re-parse from an in-memory buffer.
    """
    df = ISS._read_instructions(path)
    if df.shape[1] > 1:
        return df
    # Attempt header-merge fallback
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        # Find first data row start (line starting with digits, then comma)
        m = re.search(r"(?m)^\s*\d{2,},", text)
        if m:
            pos = m.start()
            header = text[:pos]
            data = text[pos:]
            # Remove any embedded newlines in header
            header = header.replace("\r", "").replace("\n", "")
            new_text = header + "\n" + data
            buf = io.StringIO(new_text)
            df2 = pd.read_csv(buf, dtype=str, keep_default_na=False, engine='python')
            if df2.shape[1] > 1:
                return df2
    except Exception:
        pass
    return df


def issues_summary_by_project(path: str, target: date, external_name: str, projects_filter: List[str] | None) -> Dict[str, Dict[str, Dict[str, ISS.Counts]]]:
    # Read and prep with analysis_issues helpers
    df = _read_instructions_robust(path)
    df = _drop_demo_rows(df)
    df = ISS._filter_quality(df)
    df = ISS._tag_external(df, external_name)

    # Parse raised-on dates for timeframes
    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)
    # Determine projects
    all_projects = _projects_from(df)
    if projects_filter:
        wanted = set(projects_filter)
        projects = [p for p in all_projects if p in wanted]
    else:
        projects = all_projects

    out: Dict[str, Dict[str, Dict[str, ISS.Counts]]] = {}
    for proj in projects:
        if "Location L0" in df.columns:
            proj_mask = df["Location L0"].astype(str).str.strip().eq(proj)
        elif "Project Name" in df.columns:
            proj_mask = df["Project Name"].astype(str).str.strip().eq(proj)
        else:
            continue
        sub = df[proj_mask]
        # Timeframe masks aligned to sub index
        masks = ISS._timeframe_masks(dates, target)
        cat_out: Dict[str, Dict[str, ISS.Counts]] = {}
        for category, cat_df in sub.groupby("__Category"):
            res: Dict[str, ISS.Counts] = {}
            for tf, mask in masks.items():
                mask_aligned = mask.reindex(cat_df.index, fill_value=False)
                res[tf] = ISS._count_frame(cat_df[mask_aligned])
            cat_out[category] = res
        # Ensure keys present
        for cat in ("External", "Internal"):
            if cat not in cat_out:
                cat_out[cat] = {tf: ISS.Counts(0, 0, 0) for tf in ("today", "month", "all")}
        out[proj] = cat_out
    return out


def print_combined_console(eqc: Dict[str, Dict[str, Dict[str, int]]], issues: Dict[str, Dict[str, Dict[str, ISS.Counts]]], target: date) -> None:
    projects = sorted(set(eqc.keys()) | set(issues.keys()))
    for proj in projects:
        print(f"Project: {proj}")
        # EQC block
        if proj in eqc:
            e_all = eqc[proj]["all"]; e_day = eqc[proj]["today"]; e_rows = eqc[proj].get("__rows", {"all": None, "today": None})
            print("  EQC:")
            if e_rows["all"] is not None:
                print(f"    Rows (after DEMO filter) → Total: {e_rows['all']}, Today: {e_rows['today']}")
            print(f"    Total → Pre: {e_all['Pre']}, During: {e_all['During']}, Post: {e_all['Post']}")
            print(f"    Today ({target.strftime('%d-%m-%Y')}) → Pre: {e_day['Pre']}, During: {e_day['During']}, Post: {e_day['Post']}")
        else:
            print("  EQC: (no data)")
        # Issues block (Quality)
        if proj in issues:
            print("  Issues (Quality, Safety excluded):")
            for cat in ("External", "Internal"):
                r = issues[proj][cat]
                t_today = r["today"]; t_mon = r["month"]; t_all = r["all"]
                print(f"    {cat}:")
                print(f"      Today → Total: {t_today.total}, Open: {t_today.open}, Closed: {t_today.closed}")
                print(f"      This Month → Total: {t_mon.total}, Open: {t_mon.open}, Closed: {t_mon.closed}")
                print(f"      Cumulative → Total: {t_all.total}, Open: {t_all.open}, Closed: {t_all.closed}")
        else:
            print("  Issues: (no data)")
        print()


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    target = _today(args.date)

    eqc_path = args.eqc
    issues_path = args.issues
    # Auto-detect common typos/alternates for Combined_EQC.* if not found
    if not os.path.exists(eqc_path):
        # Try .scv typo
        alt = os.path.splitext(eqc_path)[0] + ".scv"
        if os.path.exists(alt):
            eqc_path = alt
    if not os.path.exists(issues_path):
        # No common alternates; keep as-is
        pass

    # --- Auto-detect & correct swapped inputs (heuristic) ---
    def _read_head(path: str, max_cols: int = 40):
        import pandas as _pd
        for sep in ('\t', ',', None):
            try:
                if sep is None:
                    dfh = _pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine='python', nrows=5)
                else:
                    dfh = _pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine='python', nrows=5)
                return list(dfh.columns)[:max_cols]
            except Exception:
                continue
        return []
    def looks_like_eqc(cols: List[str]) -> bool:
        norm = [c.strip().lower() for c in cols]
        return ('stage' in norm and 'eqc type' in norm) or ('stage' in norm and 'eqc' in norm) or ('eqc type' in norm)
    def looks_like_instructions(cols: List[str]) -> bool:
        norm = [c.strip().lower() for c in cols]
        return any(k in norm for k in ['reference id','current status','raised on date'])
    eqc_cols = _read_head(eqc_path)
    issues_cols = _read_head(issues_path)
    if eqc_cols and issues_cols:
        eqc_is_eqc = looks_like_eqc(eqc_cols)
        issues_is_instructions = looks_like_instructions(issues_cols)
        # If the designated EQC file does NOT look like EQC but the issues file DOES, attempt swap
        if not eqc_is_eqc and looks_like_eqc(issues_cols):
            print(f"[auto-swap] The file '{eqc_path}' does not look like EQC (cols: {eqc_cols[:6]} …) but '{issues_path}' does. Swapping roles.")
            eqc_path, issues_path = issues_path, eqc_path
        # If the issues file does not look like instructions but eqc does, keep as-is.
        # If both look like EQC or both like instructions, proceed without swap (ambiguous case).

    try:
        eqc_res = eqc_summary_by_project(eqc_path, target, args.projects)
    except ValueError as ve:
        msg = str(ve)
        if "not be an EQC extract" in msg and issues_cols and looks_like_eqc(issues_cols):
            # Last chance swap if not already swapped
            print(f"[retry-swap] Detected EQC schema in issues file; retrying with swapped inputs.")
            eqc_res = eqc_summary_by_project(issues_path, target, args.projects)
            # Reassign so issues summary uses the other path
            issues_path, eqc_path = eqc_path, issues_path
        else:
            raise
    issues_res = issues_summary_by_project(issues_path, target, args.external_name, args.projects)
    print_combined_console(eqc_res, issues_res, target)


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
