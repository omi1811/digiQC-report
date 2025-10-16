import os
import re
import pandas as pd
from typing import Tuple
import argparse
from project_utils import canonicalize_project_name, canonical_project_from_row

# --- CLI args / site name ---
parser = argparse.ArgumentParser(description="EQC report generator (Weekly / Monthly / Cumulative)")
parser.add_argument("--input", "-i", default="Combined_EQC.csv", help="Input combined EQC CSV/TSV file")
parser.add_argument("--mode", "-m", choices=["weekly", "monthly", "cumulative"], help="Report mode. Default: weekly (also builds monthly & cumulative and combines to Excel when weekly is selected)")
args = parser.parse_args()
eqc_file = args.input
if not os.path.exists(eqc_file):
    # If user didn't supply a custom path (using default), try common alternates
    if args.input == "Combined_EQC.csv":
        for cand in ("Combined_EQC.csv", "Combined_EQC.scv"):
            if os.path.exists(cand):
                eqc_file = cand
                break
    # If still missing, raise a clearer error
    if not os.path.exists(eqc_file):
        raise FileNotFoundError(
            f"Input file not found. Tried: {args.input} and common alternates (Combined_EQC.csv, Combined_EQC.scv) in {os.getcwd()}"
        )

# site_name will be set per project later
site_name = None

# --- Load data (robust CSV/TSV reader) ---
def read_eqc(path: str) -> pd.DataFrame:
    last_err = None
    for sep in ('\t', ',', None):
        try:
            if sep is None:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine='python')
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine='python')
            return df
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Failed to read input file")

df = read_eqc(eqc_file)

# Keep only columns we actually use to reduce memory, everything else is dropped early
needed_cols = [
    'Date', 'Eqc Type', 'Location L0', 'Location L1', 'Location L2', 'Location L3', 'Stage', 'Status', 'Project'
]
existing_needed = [c for c in needed_cols if c in df.columns]
if existing_needed:
    df = df[existing_needed].copy()

# Drop DEMO projects entirely (rows where Project/Project Name/Location L0 contains 'DEMO')
def _drop_demo_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in
    cols = [c for c in ("Project", "Project Name", "Location L0") if c in d.columns]
    if not cols:
        return d
    mask_demo = pd.Series(False, index=d.index)
    for c in cols:
        mask_demo = mask_demo | d[c].astype(str).str.contains("DEMO", case=False, na=False)
    return d[~mask_demo]

df = _drop_demo_rows(df)

# Canonicalize project key using shared helper with fallbacks
def _canonical_project_from_row(row: pd.Series) -> str:
    return canonical_project_from_row(row)

# Derive site name from Location L0 (first non-empty value). Fallback to cwd or 'site'
loc0 = df.get('Location L0', pd.Series(dtype=str)).astype(str).str.strip()
sites = [s for s in loc0.unique() if s and s.strip()]
if sites:
    site_name = str(sites[0]).replace(' ', '_')
else:
    site_name = os.getenv('SITE') or os.path.basename(os.getcwd()) or 'site'

# Filenames are determined per project below

# Drop unwanted columns if present
drop_cols = ["Approver", "Approved Timestamp", "Time Zone",
             "Team", "Total EQC Stages", "Fail Stages", "% Fail", "URL"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Convert date (source often uses DD-MM-YYYY)
df["Date"] = pd.to_datetime(df.get("Date", ""), dayfirst=True, errors="coerce").dt.normalize()

# --- Date filter helpers ---
today = pd.Timestamp.today().normalize()

def filter_by_mode(base: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "Date" not in base.columns:
        return base.iloc[0:0].copy() if mode in ("weekly", "monthly") else base.copy()
    if mode == "weekly":
        start = today - pd.Timedelta(days=6)
        return base[(base["Date"] >= start) & (base["Date"] <= today)].copy()
    if mode == "monthly":
        month_start = today.replace(day=1)
        return base[(base["Date"] >= month_start) & (base["Date"] <= today)].copy()
    # cumulative
    return base.copy()

# --- Normalize Stage ---
def normalize_stage(stage: str) -> str:
    s = str(stage or "").lower()
    if "pre" in s:
        return "Pre"
    if "during" in s:
        return "During"
    if "post" in s:
        return "Post"
    if "reinforce" in s:
        return "Reinforcement"
    if "shutter" in s:
        return "Shuttering"
    return "Other"

def process_report(df: pd.DataFrame, site_name: str, label: str) -> None:
    if df is None or df.empty:
        # Still emit empty alerts and an empty wide CSV
        wide_filename = f"{site_name}-{label}-digiQC-report_EQC_Summary_WithTotals_Building-wise.csv"
        alerts_filename = f"{site_name}-{label}-digiQC-report_EQC_Alerts.csv"
        pd.DataFrame([], columns=["Alerts"]).to_csv(alerts_filename, index=False)
        pd.DataFrame(columns=["Checklists"]).to_csv(wide_filename, index=False)
        print(f"✅ Wrote empty {label} outputs for {site_name} (no rows in window)")
        return
    # Normalize Stage
    df["Stage_Norm"] = df.get("Stage", "").apply(normalize_stage)
    # Stage order
    stage_order = ["Pre", "During", "Post", "Reinforcement", "Shuttering", "Other"]
    df["Stage_Norm"] = pd.Categorical(df["Stage_Norm"], categories=stage_order, ordered=True)
    # Reduce memory: convert key columns to category dtype
    for col in ["Eqc Type", "Location L1", "Location L2", "Location L3", "Status"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Pivot
    print(f"➡️  Building pivot for {site_name} ({label}) on {len(df)} rows…")
    summary = (
        df.groupby(["Eqc Type", "Location L1", "Stage_Norm"], observed=True)
          .size()
          .unstack(level=[1, 2], fill_value=0)
    )
    # Per-building total
    for bldg in df.get("Location L1", pd.Series(dtype=str)).unique():
        if not bldg:
            continue
        try:
            if all(((bldg, stage) in summary.columns) for stage in ["Pre", "During", "Post"]):
                summary[(bldg, "Total")] = (
                    summary[(bldg, "Pre")] + summary[(bldg, "During")] + summary[(bldg, "Post")]
                )
        except Exception:
            continue
    # Overall totals
    totals = (
        df.groupby(["Eqc Type", "Stage_Norm"], observed=True)
          .size()
          .unstack(fill_value=0)
    )
    totals.columns = pd.MultiIndex.from_product([["Total"], totals.columns.astype(str)])
    summary = pd.concat([summary, totals], axis=1)
    summary.index.name = "EQC Checklist"
    # Note: We intentionally skip adding a Status-wide summary into the main pivot to reduce memory/time.
    # Alerts are still computed separately below.
    # Alerts
    alerts = []
    redo_col = df.get("Status", pd.Series(["" for _ in range(len(df))], index=df.index)).astype(str).str.upper()
    redo_df = df[redo_col == "REDO"]
    if not redo_df.empty:
        max_dates = (
            redo_df.groupby(["Eqc Type", "Location L1", "Location L2", "Location L3"], observed=True)["Date"].max()
        )
        for keys, last_date in max_dates.items():
            eqc = keys[0]
            loc_parts = [k for k in keys[1:] if k and str(k).strip()]
            loc_label = " / ".join(loc_parts) if loc_parts else (keys[1] or "UNKNOWN")
            if pd.notna(last_date) and (today - last_date).days > 3:
                alerts.append(f"ALERT: {eqc} in {loc_label} is REDO for more than 3 days (last: {last_date.date()})")
    status_col = df.get("Status", pd.Series(["" for _ in range(len(df))], index=df.index)).astype(str).str.upper()
    inprog_df = df[status_col == "IN_PROGRESS"]
    if not inprog_df.empty:
        min_dates = (
            inprog_df.groupby(["Eqc Type", "Location L1", "Location L2", "Location L3", "Stage_Norm"], observed=True)["Date"].min()
        )
        for keys, first_date in min_dates.items():
            eqc = keys[0]
            stage = keys[4]
            loc_parts = [k for k in keys[1:4] if k and str(k).strip()]
            loc_label = " / ".join(loc_parts) if loc_parts else (keys[1] or "UNKNOWN")
            if pd.notna(first_date) and (today - first_date).days > 21:
                alerts.append(f"ALERT: {eqc} in {loc_label} stuck in {stage} for more than 3 weeks (since: {first_date.date()})")
    # File names for this project
    wide_filename = f"{site_name}-{label}-digiQC-report_EQC_Summary_WithTotals_Building-wise.csv"
    alerts_filename = f"{site_name}-{label}-digiQC-report_EQC_Alerts.csv"
    pd.DataFrame(alerts, columns=["Alerts"]).to_csv(alerts_filename, index=False)
    print(f"✅ Alerts written to {alerts_filename}")
    # Prepare wide-format CSV
    print(f"Raw entries ({label}):", len(df))
    print(f"Summary total count ({label}):", summary.select_dtypes(include=["number"]).sum().sum())
    summary.columns = pd.MultiIndex.from_tuples([c if isinstance(c, tuple) else ("", str(c)) for c in summary.columns])
    building_totals = {}
    for bldg in sorted({c[0] for c in summary.columns if c[0] and c[0] not in ("Total", "Status")}):
        cols = [c for c in summary.columns if c[0] == bldg and c[1] != "Total"]
        if cols:
            building_totals[(bldg, "Total")] = summary.loc[:, cols].sum(axis=1)
    if building_totals:
        bt_df = pd.concat(building_totals, axis=1)
        summary = pd.concat([summary, bt_df], axis=1)
    numeric_sum = summary.select_dtypes(include=["number"]).sum()
    if not numeric_sum.empty:
        grand_total = numeric_sum.to_frame().T
        grand_total.index = ["TOTAL"]
        summary_final = pd.concat([summary, grand_total], axis=0, sort=False)
    else:
        summary_final = summary.copy()
    stage_rank = {s: i for i, s in enumerate(stage_order)}
    def sort_key(col: Tuple[str, str]):
        building, stage = col
        if building == "Total":
            return (2, "ZZZ", stage_rank.get(stage, 99), str(stage))
        if building == "":
            return (3, "ZZZ", 99, str(stage))
        return (1, building, stage_rank.get(stage, 99), str(stage))
    if isinstance(summary_final.columns, pd.MultiIndex) and not summary_final.columns.is_unique:
        # Avoid deprecated axis=1 groupby by transposing
        summary_final = summary_final.T.groupby(level=list(range(summary_final.columns.nlevels))).sum().T
    sorted_cols = sorted(summary_final.columns.tolist(), key=sort_key)
    summary_final = summary_final.reindex(columns=sorted_cols)
    print(f"✅ Computed {label} combined summary in memory; writing only building-wise wide report next")
    # Build tidy per-building -> wide for output CSV
    try:
        raw_buildings = list(df.get('Location L1', pd.Series(dtype=str)).unique())
        buildings = sorted({(b if b else 'UNKNOWN') for b in raw_buildings})
        rows = []
        for bldg in buildings:
            for eqc in summary_final.index:
                if str(eqc).upper() == 'TOTAL':
                    continue
                row = {"EQC Checklist": eqc, "Building": bldg}
                for s in stage_order:
                    val = 0
                    lookup_building = '' if bldg == 'UNKNOWN' else bldg
                    lookup_col = (lookup_building, s)
                    if lookup_col in summary_final.columns:
                        try:
                            val = int(summary_final.at[eqc, lookup_col])
                        except Exception:
                            try:
                                val = int(summary_final.loc[eqc, lookup_col])
                            except Exception:
                                val = 0
                    row[s] = val
                row['Total Count'] = sum(row[s] for s in stage_order)
                rows.append(row)
        long_df = pd.DataFrame(rows)
        if long_df.empty:
            print('No weekly per-building rows to write.')
            # Still write an empty CSV with header
            pd.DataFrame(columns=["Checklists"]).to_csv(wide_filename, index=False)
            print(f"✅ Wrote weekly wide-format per-building summary to {wide_filename}")
            return
        def longest_common_prefix_tokens(names):
            if not names:
                return ''
            token_lists = [str(n).split() for n in names]
            prefix = []
            for tokens in zip(*token_lists):
                if all(t == tokens[0] for t in tokens):
                    prefix.append(tokens[0])
                else:
                    break
            return ' '.join(prefix).strip()
        def base_name(n: str) -> str:
            if not isinstance(n, str):
                return str(n)
            s = n.strip()
            s_norm = s.lower()
            fixes = {
                r'painting.*internal': 'Painting Works : Internal',
                r'painting.*external': 'Painting Works : External',
                r'painting\s*works\s*:\s*internal': 'Painting Works : Internal',
                r'painting\s*works\s*:\s*external': 'Painting Works : External',
                r'waterproof.*boxtype': 'Waterproofing works: Toilet and Skirting',
                r'waterproof.*toilet': 'Waterproofing works: Toilet and Skirting',
                r'waterproof.*skirting': 'Waterproofing works: Toilet and Skirting',
                r'tiling.*kitchen.*platform': 'Tiling - Kitchen Platform',
                r'tiling.*kitchen.*sink': 'Tiling - Kitchen Platform',
                r'tiling[-\s]*toilet.*dado': 'Tiling - Toilet Dado',
                r'kitchen\s*dado': 'Kitchen Dado Checklist',
            }
            for pat, name in fixes.items():
                if re.search(pat, s_norm):
                    return name
            if '.' in s:
                return s.split('.', 1)[0].strip()
            if ':' in s:
                return s.split(':', 1)[0].strip()
            tokens = s.split()
            if len(tokens) >= 2:
                key = ' '.join(tokens[:2])
                same_prefix = [x for x in summary_final.index.astype(str) if x.startswith(key)]
                if len(same_prefix) > 1:
                    lcp = longest_common_prefix_tokens(same_prefix)
                    return lcp if lcp else key
            return ' '.join(tokens[:4]).strip()
        long_df['Checklists'] = long_df['EQC Checklist'].apply(base_name)
        agg = long_df.groupby(['Checklists', 'Building'])[stage_order].sum().reset_index()
        for s in stage_order:
            if s in agg.columns:
                agg[s] = pd.to_numeric(agg[s], errors='coerce').fillna(0).astype(int)
        # New cumulative logic per request across all files:
        # Pre = Pre + During + Post + Reinforcement + Shuttering + Other
        # During = During + Post + Other
        # Post = Post + Other
        if all(col in agg.columns for col in ['Pre', 'During', 'Post']):
            pre_components = [agg.get(x, 0) for x in ['Pre', 'During', 'Post']]
            pre_components += [agg.get(x, 0) for x in ['Reinforcement', 'Shuttering', 'Other']]
            agg['Pre'] = sum(pre_components)
            agg['During'] = agg.get('During', 0) + agg.get('Post', 0) + agg.get('Other', 0)
            agg['Post'] = agg.get('Post', 0) + agg.get('Other', 0)
        # Total Count based only on Pre/During/Post to avoid double counting helper stages
        total_stage_cols = [c for c in ['Pre', 'During', 'Post'] if c in agg.columns]
        if total_stage_cols:
            agg['Total Count'] = agg[total_stage_cols].sum(axis=1)
        else:
            agg['Total Count'] = 0
        # Build overall using only existing main stage columns
        stage_present = [s for s in ['Pre', 'During', 'Post'] if s in agg.columns]
        if not stage_present:
            stage_present = [s for s in stage_order if s in agg.columns]
        overall = agg.groupby('Checklists')[stage_present].sum().reset_index()
        overall['Building'] = 'ALL'
        if total_stage_cols:
            overall['Total Count'] = overall[total_stage_cols].sum(axis=1)
        else:
            overall['Total Count'] = 0
        final_by_building = pd.concat([agg, overall], ignore_index=True, sort=False)
        # Arrange columns based on what is present
        cols_order = ['Checklists', 'Building'] + stage_present + ['Total Count']
        final_by_building = final_by_building[cols_order]
        # Restrict wide view to main stages only
        stages_for_wide = [c for c in ['Pre', 'During', 'Post'] if c in final_by_building.columns]
        wide_idx = final_by_building.set_index(['Checklists', 'Building'])[stages_for_wide]
        wide_un = wide_idx.unstack(level='Building', fill_value=0)
        # columns are (Stage, Building); swap to (Building, Stage)
        wide_un.columns = wide_un.columns.swaplevel(0, 1)
        buildings_set = sorted({c[0] for c in wide_un.columns})
        for b in buildings_set:
            stage_cols = [(b, s) for s in ['Pre', 'During', 'Post'] if (b, s) in wide_un.columns]
            if stage_cols:
                wide_un[(b, 'Total')] = wide_un[stage_cols].sum(axis=1)
        buildings = sorted({c[0] for c in wide_un.columns})
        if 'ALL' in buildings:
            buildings.remove('ALL')
            buildings.append('ALL')
        ordered_cols = []
        for b in buildings:
            for s in ['Pre', 'During', 'Post', 'Total']:
                if (b, s) in wide_un.columns:
                    ordered_cols.append((b, s))
        wide_un = wide_un.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
        # Do NOT append a TOTAL row to avoid accidental double counting when users sum the column.
        # Drop per-building Total before flattening
        if isinstance(wide_un.columns, pd.MultiIndex):
            cols_to_keep = [c for c in wide_un.columns if str(c[1]) != 'Total']
            wide_un = wide_un.loc[:, cols_to_keep]
        # Flatten MultiIndex columns to single header names like 'Building - Pre'
        flat_cols = [f"{b} - {s}" for (b, s) in wide_un.columns]
        wide_un.columns = flat_cols
        wide_un_reset = wide_un.reset_index()
        # Keep all checklist rows (including numeric-only names) to preserve totals parity with dashboard
        wide_un_reset.to_csv(wide_filename, index=False)
        print(f"✅ Wrote {label} wide-format per-building summary to {wide_filename}")
    except Exception as e:
        print(f'Failed to write {label} wide-format per-building summary:', str(e))

# --- Determine mode (non-interactive): default to weekly ---
mode = args.mode or "weekly"

# --- Split by project and process ---
# Build canonical project keys
df["__ProjectKey"] = df.apply(_canonical_project_from_row, axis=1)
projects = [p for p in df["__ProjectKey"].astype(str).str.strip().unique() if p and 'demo' not in p.lower()]

def generate_for_mode(run_mode: str) -> None:
    win = filter_by_mode(df, run_mode)
    lbl = "Weekly" if run_mode == "weekly" else ("Monthly" if run_mode == "monthly" else "Cumulative")
    for proj in projects:
        sub_win = win.copy()
        sub_win["__ProjectKey"] = sub_win.apply(_canonical_project_from_row, axis=1)
        sub = sub_win[sub_win["__ProjectKey"].astype(str).str.strip() == proj].copy()
        site = str(proj).replace(' ', '_')
        process_report(sub, site, lbl)

# If weekly is selected (or defaulted via menu), generate all three modes to ensure the combined workbook has everything
if mode == "weekly":
    for m in ("weekly", "monthly", "cumulative"):
        generate_for_mode(m)
else:
    generate_for_mode(mode)

# Always build the combined workbook after generation so the user gets an up-to-date Excel
try:
    import combine_reports_to_excel as comb
    comb.main()
except Exception as e:
    print(f"Failed to build combined workbook automatically: {e}. You can run 'python3 combine_reports_to_excel.py' manually.")
