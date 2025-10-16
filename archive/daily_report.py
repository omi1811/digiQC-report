#!/usr/bin/env python3
"""Generate daily EQC report (per-project CSVs and combined Excel workbook).

Usage: python3 daily_report.py [--date DD-MM-YYYY] [--eqc PATH]
If --date is omitted, today's date is used.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, date
import pandas as pd


def read_eqc(path: str) -> pd.DataFrame:
    last_err = None
    for sep in ("\t", ",", None):
        try:
            if sep is None:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine='python')
            else:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine='python')
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Failed to read input file")


def canonical_project_from_row(row: pd.Series) -> str:
    l0 = str(row.get("Location L0", "") or "").strip()
    if l0:
        return l0
    proj = str(row.get("Project", "") or "").strip()
    m = {
        "Itrend Futura": "FUTURA",
        "FUTURA": "FUTURA",
        "City Life": "Itrend City Life",
        "Itrend City Life": "Itrend City Life",
        "Itrend Palacio": "Itrend-Palacio",
        "Itrend-Palacio": "Itrend-Palacio",
    }
    return m.get(proj, proj)


def drop_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Project", "Project Name", "Location L0") if c in df.columns]
    if not cols:
        return df
    mask_demo = pd.Series(False, index=df.index)
    for c in cols:
        mask_demo = mask_demo | df[c].astype(str).str.contains("DEMO", case=False, na=False)
    return df[~mask_demo]


def normalize_stage(stage: str) -> str:
    s = str(stage or "").lower()
    if "pre" in s:
        return "Pre"
    if "during" in s:
        return "During"
    if "post" in s:
        return "Post"
    return "Other"


def build_daily_reports(eqc_path: str, target_date: date, out_xlsx: str = None) -> list[str]:
    df = read_eqc(eqc_path)
    df = drop_demo_rows(df)
    # normalize
    df["Date"] = pd.to_datetime(df.get("Date", ""), dayfirst=True, errors='coerce').dt.normalize()
    df["__ProjectKey"] = df.apply(canonical_project_from_row, axis=1)
    df["Stage_Norm"] = df.get("Stage", "").map(normalize_stage)

    projects = [p for p in sorted(df['__ProjectKey'].astype(str).unique()) if p and 'demo' not in p.lower()]
    produced = []

    # Filter to the specific day
    day_df = df[df['Date'] == pd.to_datetime(target_date).normalize()].copy()

    # For alerts we will use all data for the project (not only today's rows)
    for proj in projects:
        sub_day = day_df[day_df['__ProjectKey'] == proj].copy()
        sub_all = df[df['__ProjectKey'] == proj].copy()
        site_name = str(proj).replace(' ', '_')

        # Build per-building per-checklist counts for the day
        rows = []
        buildings = sorted({b if (b and str(b).strip()) else 'UNKNOWN' for b in sub_day.get('Location L1', pd.Series(dtype=str)).unique()})
        if not buildings:
            buildings = ['UNKNOWN']
        checklists = sorted(sub_day.get('Eqc Type', pd.Series(dtype=str)).fillna('').unique())
        for bldg in buildings:
            for eqc in checklists:
                mask = (sub_day.get('Eqc Type', '') == eqc) & (sub_day.get('Location L1', '').fillna('') == ('' if bldg == 'UNKNOWN' else bldg))
                if not mask.any():
                    continue
                pre = int(((sub_day[mask]['Stage_Norm'] == 'Pre')).sum())
                during = int(((sub_day[mask]['Stage_Norm'] == 'During')).sum())
                post = int(((sub_day[mask]['Stage_Norm'] == 'Post')).sum())
                total = pre + during + post
                rows.append({
                    'EQC Checklist': eqc,
                    'Building': bldg,
                    'Pre': pre,
                    'During': during,
                    'Post': post,
                    'Total Count': total,
                })

        df_out = pd.DataFrame(rows)
        # ensure columns exist even if empty
        for c in ['Pre', 'During', 'Post', 'Total Count']:
            if c not in df_out.columns:
                df_out[c] = 0

        # Write per-project CSV
        csv_name = f"{site_name}-Daily-digiQC-report_EQC_Summary_WithTotals_Building-wise.csv"
        df_out.to_csv(csv_name, index=False)
        produced.append(csv_name)
        print(f"Wrote daily CSV for {proj}: {csv_name} rows:{len(df_out)}")

    # Write combined workbook
    if out_xlsx is None:
        out_xlsx = f"EQC_Daily_Report_{target_date.strftime('%Y-%m-%d')}_AllProjects.xlsx"
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for csv in produced:
            try:
                dfp = pd.read_csv(csv)
                sheet_name = os.path.basename(csv).split('-Daily-')[0][:31]
                dfp.to_excel(writer, sheet_name=sheet_name + ' Daily', index=False)
            except Exception:
                continue
    print('Wrote combined workbook', out_xlsx)
    produced.append(out_xlsx)
    return produced


def main():
    p = argparse.ArgumentParser(description='Daily EQC report generator')
    p.add_argument('--date', '-d', help='Date for the daily report in DD-MM-YYYY format (default: today)')
    p.add_argument('--eqc', default='Combined_EQC.csv', help='Path to combined EQC file')
    args = p.parse_args()

    eqc_path = args.eqc
    if not os.path.exists(eqc_path):
        alt = os.path.splitext(eqc_path)[0] + '.scv'
        if os.path.exists(alt):
            eqc_path = alt
    if not os.path.exists(eqc_path):
        raise SystemExit(f'Combined EQC file not found: tried {args.eqc} and alternates')

    if args.date:
        try:
            target = datetime.strptime(args.date, '%d-%m-%Y').date()
        except Exception:
            raise SystemExit('Date must be in DD-MM-YYYY format')
    else:
        target = date.today()

    build_daily_reports(eqc_path, target)


if __name__ == '__main__':
    main()
