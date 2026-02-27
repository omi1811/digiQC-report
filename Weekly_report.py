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

# Determine if user explicitly provided a file (not using default)
explicit_file_provided = (args.input != "Combined_EQC.csv")

print(f"=" * 80)
print(f"Input file argument: {args.input}")
print(f"Explicit file provided: {explicit_file_provided}")
print(f"=" * 80)

# First, if a specific file was provided and exists, verify it's an EQC file
if os.path.exists(eqc_file):
    try:
        # Try reading with different separators to handle TSV/CSV
        # Try comma first (most common), then tab, then auto-detect
        test_df = None
        last_error = None
        for sep in (',', '\t', None):
            try:
                if sep is None:
                    test_df = pd.read_csv(eqc_file, nrows=1, dtype=str, keep_default_na=False, sep=None, engine='python')
                else:
                    test_df = pd.read_csv(eqc_file, nrows=1, dtype=str, keep_default_na=False, sep=sep, engine='python')
                
                # Verify we got actual columns (not just one column with all data)
                if len(test_df.columns) > 5:  # EQC files should have many columns
                    break
                else:
                    # This separator created too few columns, try next one
                    test_df = None
                    continue
            except Exception as e:
                last_error = e
                continue
        
        if test_df is None:
            raise Exception(f"Unable to parse file with any separator. Last error: {last_error}")
        
        # Verify it has EQC columns
        if "Eqc Type" not in test_df.columns and "Stage" not in test_df.columns:
            print(f"\n[ERROR] {eqc_file} is NOT an EQC file!")
            print(f"Available columns: {', '.join(test_df.columns.tolist()[:10])}")
            # Check if it's an Issues file
            if "Reference ID" in test_df.columns or "Type L0" in test_df.columns:
                print("This appears to be an ISSUES file, not an EQC file!")
            
            # If explicit file was provided (e.g., via web upload), DO NOT auto-detect
            if explicit_file_provided:
                print("\n" + "=" * 80)
                print("CRITICAL ERROR: Uploaded file is not an EQC file!")
                print("EQC files must have columns: 'Eqc Type', 'Stage', 'Status', 'Date'")
                print("Issues files have columns: 'Reference ID', 'Type L0', 'Raised By'")
                print("Please upload the correct EQC export file.")
                print("=" * 80)
                raise ValueError(f"Invalid file type. Expected EQC file, got: {', '.join(test_df.columns.tolist()[:5])}")
            else:
                print("Attempting auto-detection of EQC file...")
                eqc_file = ""  # Force auto-detection only if no explicit file
        else:
            print(f"\n[OK] VALIDATED: Using uploaded EQC file: {eqc_file}")
            print(f"  Columns found: Eqc Type={('Eqc Type' in test_df.columns)}, Stage={('Stage' in test_df.columns)}")
    except Exception as e:
        print(f"\n[ERROR] reading {eqc_file}: {e}")
        
        # If explicit file was provided, FAIL instead of auto-detecting
        if explicit_file_provided:
            print("\n" + "=" * 80)
            print("CRITICAL ERROR: Cannot read uploaded file!")
            print(f"File: {eqc_file}")
            print(f"Error: {e}")
            print("=" * 80)
            raise
        else:
            print("Attempting auto-detection of EQC file...")
            eqc_file = ""  # Force auto-detection only if no explicit file

# If file doesn't exist or validation failed, try auto-detection
# BUT only if no explicit file was provided (i.e., using default)
if (not eqc_file or not os.path.exists(eqc_file)) and not explicit_file_provided:
    print("\nAuto-detecting EQC file...")
    # Try common patterns in current directory
    candidates = [
        "Combined_EQC.csv", 
        "Combined_EQC.scv",
        "EQC.csv",
        "eqc.csv"
    ]
    
    # Check current directory for EQC-*.csv patterns
    try:
        for f in os.listdir(os.getcwd()):
            if (f.startswith("EQC-") or f.startswith("eqc-") or f.startswith("EQC_")) and f.endswith(".csv"):
                candidates.insert(0, f)
    except:
        pass
    
    # Also check parent directory for EQC files
    parent_dir = os.path.dirname(os.getcwd())
    if parent_dir and os.path.exists(parent_dir):
        try:
            for f in os.listdir(parent_dir):
                if (f.startswith("EQC-") or f.startswith("eqc-") or f.startswith("EQC_")) and f.endswith(".csv"):
                    candidates.insert(0, os.path.join(parent_dir, f))
        except:
            pass
    
    for cand in candidates:
        if os.path.exists(cand):
            # Verify it's an EQC file by checking for required columns
            try:
                # Try reading with different separators
                test_df = None
                for sep in ('\t', ',', None):
                    try:
                        if sep is None:
                            test_df = pd.read_csv(cand, nrows=1, dtype=str, keep_default_na=False, sep=None, engine='python')
                        else:
                            test_df = pd.read_csv(cand, nrows=1, dtype=str, keep_default_na=False, sep=sep, engine='python')
                        break
                    except:
                        continue
                
                if test_df is None:
                    continue
                
                # EQC files should have these columns
                if "Eqc Type" in test_df.columns or "Stage" in test_df.columns:
                    eqc_file = cand
                    print(f"Found EQC file: {cand}")
                    break
            except:
                continue
    
    # If still missing, raise a clearer error with helpful info
    if not eqc_file or not os.path.exists(eqc_file):
        # Try to find any CSV files to help the user
        available_csvs = []
        try:
            for f in os.listdir(os.getcwd()):
                if f.endswith(".csv"):
                    available_csvs.append(f)
        except:
            pass
        parent_dir = os.path.dirname(os.getcwd())
        if parent_dir and os.path.exists(parent_dir):
            try:
                for f in os.listdir(parent_dir):
                    if f.endswith(".csv"):
                        available_csvs.append(f"../{f}")
            except:
                pass
        
        error_msg = f"Input file not found: {args.input}\n\n"
        if available_csvs:
            error_msg += f"Available CSV files:\n"
            for csv in available_csvs[:10]:
                error_msg += f"  - {csv}\n"
            error_msg += f"\nRun: python identify_csv_files.py to identify which is the EQC file"
        else:
            error_msg += "No CSV files found in current or parent directory."
        error_msg += f"\n\nEQC files should have columns: Eqc Type, Stage, Inspector, Status, Date"
        raise FileNotFoundError(error_msg)
elif explicit_file_provided and not os.path.exists(eqc_file):
    # Explicit file was provided but doesn't exist
    raise FileNotFoundError(f"Specified file does not exist: {eqc_file}")

# Final confirmation of which file is being used
print("\n" + "=" * 80)
print(f"FINAL: Processing EQC file: {eqc_file}")
print(f"File size: {os.path.getsize(eqc_file) / 1024:.1f} KB")
print("=" * 80 + "\n")

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

print(f"\n[DEBUG] Loaded {len(df)} rows from file")
print(f"[DEBUG] Columns: {df.columns.tolist()}")

# Keep only columns we actually use to reduce memory, everything else is dropped early
needed_cols = [
    'Date', 'Eqc Type', 'Location L0', 'Location L1', 'Location L2', 'Location Variable', 'Stage', 'Status', 'Project'
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

print(f"[DEBUG] After DEMO filter: {len(df)} rows")
if len(df) == 0:
    print("\n" + "=" * 80)
    print("WARNING: All rows were filtered out (possibly all DEMO projects)")
    print("=" * 80)

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
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
else:
    df["Date"] = pd.NaT

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
    # Single-stage checklists - these should appear in all columns (Pre, During, Post)
    # Check these FIRST before general keywords to avoid miscategorization
    single_stage_patterns = [
        "pressure test", "pressure testing",
        "paver block", "paver",
        "kerb stone", "kerbstone", "curb stone",
        "water test", "leak test",
        "hydrostatic test",
    ]
    for pattern in single_stage_patterns:
        if pattern in s:
            return "Other"
    
    # Standard mappings - check these after single-stage patterns
    # Pre/Before variations - check first
    if "pre" in s or "before" in s or "prior" in s:
        return "Pre"
    # Post/After variations - check before 'during' to handle "POST FIXING" correctly
    if "post" in s or "after" in s:
        return "Post"
    # During/Fixing variations - check after Pre/Post
    # Exclude single-stage patterns that contain "fixing"
    if "during" in s or "internal plumbing" in s or "internal handover" in s:
        return "During"
    # Only categorize as During if "fixing" is part of a multi-stage checklist
    if "fixing" in s and ("pre" in s or "post" in s or "during" in s):
        return "During"
    # Pour card variations
    if "pour card" in s or "pour-card" in s:
        # If we got here, no pre/during/post was found, default to During
        return "During"
    # Reinforcement variations
    if "reinforce" in s:
        return "Reinforcement"
    # Shuttering variations
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
        print(f"[OK] Wrote empty {label} outputs for {site_name} (no rows in window)")
        return
    # Normalize Stage
    df["Stage_Norm"] = df.get("Stage", "").apply(normalize_stage)
    # Stage order
    stage_order = ["Pre", "During", "Post", "Reinforcement", "Shuttering", "Other"]
    df["Stage_Norm"] = pd.Categorical(df["Stage_Norm"], categories=stage_order, ordered=True)
    # Reduce memory: convert key columns to category dtype
    for col in ["Eqc Type", "Location L1", "Location L2", "Location Variable", "Status"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Pivot
    print(f"-> Building pivot for {site_name} ({label}) on {len(df)} rows...")
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
            redo_df.groupby(["Eqc Type", "Location L1", "Location L2", "Location Variable"], observed=True)["Date"].max()
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
            inprog_df.groupby(["Eqc Type", "Location L1", "Location L2", "Location Variable", "Stage_Norm"], observed=True)["Date"].min()
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
    print(f"[OK] Alerts written to {alerts_filename}")
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
    print(f"[OK] Computed {label} combined summary in memory; writing only building-wise wide report next")
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
            print(f"[OK] Wrote weekly wide-format per-building summary to {wide_filename}")
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
            
            # Handle "Checklist for ..." names - preserve full name with Title Case
            if s_norm.startswith('checklist for '):
                # Remove contractor suffix if present (double space + name pattern)
                cleaned = re.sub(r'\s{2,}[A-Za-z]+\s*$', '', s)
                cleaned = re.sub(r'\s+\([^)]+\)\s*$', '', cleaned)  # Remove (Contractor Name)
                return cleaned.strip().title()
            
            # Handle RCC checklists - preserve name as-is, just remove contractor prefix
            if s_norm.startswith('rcc ') or s_norm.startswith('rcc-') or re.match(r'^rcc\s*\(', s_norm):
                # Remove contractor suffix if present (double space + name pattern)
                cleaned = re.sub(r'\s{2,}[A-Za-z]+\s*$', '', s)
                cleaned = re.sub(r'\s+\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\)\s*$', '', cleaned)  # Remove (Contractor Name)
                # Remove leading contractor name before RCC if present (e.g., "ContractorName RCC ...")
                cleaned = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?=RCC)', '', cleaned, flags=re.IGNORECASE)
                return cleaned.strip().title()
            
            # Pattern-based fixes with standardized names
            fixes = {
                r'painting.*internal': 'Painting Works : Internal',
                r'painting.*external': 'Painting Works : External',
                r'painting\s*works\s*:\s*external\s*texture': 'Painting Works : External Texture',
                r'waterproof.*boxtype': 'Waterproofing Works: Toilet and Skirting',
                r'waterproof.*toilet': 'Waterproofing Works: Toilet and Skirting',
                r'waterproof.*skirting': 'Waterproofing Works: Toilet and Skirting',
                r'tiling.*kitchen.*platform': 'Tiling - Kitchen Platform',
                r'tiling.*kitchen.*sink': 'Tiling - Kitchen Platform',
                r'tiling[-\s]*toilet.*dado': 'Tiling - Toilet Dado',
                r'tiling.*flooring': 'Tiling - Flooring Work',
                r'kitchen\s*dado': 'Kitchen Dado',
                r'gypsum\s*plaster': 'Gypsum Plaster Works',
                r'aac\s*block': 'AAC Block Work',
                r'brick\s*work': 'Brick Work',
                r'external\s*plaster': 'External Plaster Works',
                r'internal\s*plaster': 'Internal Plaster Works',
                r'm\s*s\s*railing': 'MS Railing',
                r's\s*s\s*railing': 'SS Railing',
                r'alluminium\s*window': 'Aluminium Window and Door Work',
                r'aluminium\s*window': 'Aluminium Window and Door Work',
                r'carpentry\s*work': 'Carpentry Work : Door Frame & Shutters',
                r'plumbing\s*work': 'Plumbing Work - Internal',
                r'electrical\s*work': 'Electrical Works',
                r'false\s*ceiling': 'False Ceiling Work',
                r'trimix': 'Trimix Work',
                r'aluform.*checklist': 'Aluform Checklist',
                r'fire\s*fighting': 'Fire Fighting Work',
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
        # Apply cumulative roll-up ONLY when label == 'Cumulative'
        if label == 'Cumulative' and all(col in agg.columns for col in ['Pre', 'During', 'Post']):
            # Other (single-stage checklists) are included in all three columns
            pre_components = [agg.get(x, 0) for x in ['Pre', 'During', 'Post', 'Reinforcement', 'Shuttering', 'Other']]
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
        # Append a TOTAL row at the end summing across all checklists
        try:
            total_row = {}
            for col in wide_un_reset.columns:
                if str(col).strip().lower() == 'checklists':
                    total_row[col] = 'TOTAL'
                else:
                    total_row[col] = pd.to_numeric(wide_un_reset[col], errors='coerce').fillna(0).astype(int).sum()
            wide_with_total = pd.concat([wide_un_reset, pd.DataFrame([total_row])], ignore_index=True)
        except Exception:
            # Fallback: write without total row if anything unexpected happens
            wide_with_total = wide_un_reset
        wide_with_total.to_csv(wide_filename, index=False)
        print(f"[OK] Wrote {label} wide-format per-building summary to {wide_filename}")
    except Exception as e:
        print(f'Failed to write {label} wide-format per-building summary:', str(e))

# --- Determine mode (non-interactive): default to weekly ---
mode = args.mode or "weekly"

# --- Split by project and process ---
# Build canonical project keys
df["__ProjectKey"] = df.apply(_canonical_project_from_row, axis=1)

# Debug: Show what projects were found
all_projects = df["__ProjectKey"].astype(str).str.strip().unique()
projects = [p for p in all_projects if p and 'demo' not in p.lower()]

print(f"\n[DEBUG] All unique projects found: {all_projects.tolist()}")
print(f"[DEBUG] Projects after filtering DEMOs: {projects}")
print(f"[DEBUG] Total projects to process: {len(projects)}\n")

if len(projects) == 0:
    print("\n" + "=" * 80)
    print("ERROR: No valid projects found in the uploaded file!")
    print("This could happen if:")
    print("  1. All projects are marked as DEMO")
    print("  2. The 'Project' column is missing or empty")
    print("  3. The file has no data rows")
    print("Continuing with empty output...")
    print("=" * 80)

def generate_for_mode(run_mode: str) -> None:
    win = filter_by_mode(df, run_mode)
    lbl = "Weekly" if run_mode == "weekly" else ("Monthly" if run_mode == "monthly" else "Cumulative")
    for proj in projects:
        sub_win = win.copy()
        sub_win["__ProjectKey"] = sub_win.apply(_canonical_project_from_row, axis=1)
        sub = sub_win[sub_win["__ProjectKey"].astype(str).str.strip() == proj].copy()
        site = str(proj).replace(' ', '_')
        process_report(sub, site, lbl)

# NOTE: Changed to generate only Cumulative mode per user request (removed Weekly/Monthly)
# The mode argument is kept for backwards compatibility but only cumulative is generated

# Clean up old CSV report files to prevent combining with stale data
# This ensures only the current run's data is included in the final Excel
print("\nCleaning up old report CSV files...")
import glob
for old_csv in glob.glob("*-Cumulative-digiQC-report_EQC_*.csv"):
    try:
        os.remove(old_csv)
        print(f"  Removed: {old_csv}")
    except Exception as e:
        print(f"  Warning: Could not remove {old_csv}: {e}")

generate_for_mode("cumulative")

# Always build the combined workbook after generation so the user gets an up-to-date Excel
try:
    import combine_reports_to_excel as comb
    comb.main()
except Exception as e:
    print(f"Failed to build combined workbook automatically: {e}. You can run 'python3 combine_reports_to_excel.py' manually.")

