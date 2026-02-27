import os
import pandas as pd
from typing import Tuple

# --- Input file ---
eqc_file = "abc.csv"

# --- Load data ---
# The source `abc.csv` is tab-separated. Specify sep='\t' to avoid tokenization errors
df = pd.read_csv(eqc_file, dtype=str, keep_default_na=False, sep='\t')

# Drop unwanted columns if present
drop_cols = ["Approver", "Approved Timestamp", "Time Zone",
             "Team", "Total EQC Stages", "Fail Stages", "% Fail", "URL"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Convert date (source uses DD-MM-YYYY)
df["Date"] = pd.to_datetime(df.get("Date", ""), dayfirst=True, errors="coerce")

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

df["Stage_Norm"] = df.get("Stage", "").apply(normalize_stage)

# --- Define custom stage order ---
stage_order = ["Pre", "During", "Post", "Reinforcement", "Shuttering", "Other"]
df["Stage_Norm"] = pd.Categorical(df["Stage_Norm"], categories=stage_order, ordered=True)

# --- Pivot table (Eqc Type as index, Building + Stage as columns) ---
# Use groupby + unstack to avoid pandas grouper errors
summary = (
    df.groupby(["Eqc Type", "Location L1", "Stage_Norm"])["Eqc Type"]
    .count()
    .unstack(level=[1, 2], fill_value=0)
)

# --- Add per-building totals (Pre+During+Post only) ---
for bldg in df.get("Location L1", pd.Series(dtype=str)).unique():
    if not bldg:
        continue
    try:
        if all(((bldg, stage) in summary.columns) for stage in ["Pre", "During", "Post"]):
            summary[(bldg, "Total")] = (
                summary[(bldg, "Pre")] +
                summary[(bldg, "During")] +
                summary[(bldg, "Post")]
            )
    except Exception:
        continue

# --- Add overall totals across buildings ---
totals = (
    df.groupby(["Eqc Type", "Stage_Norm"])["Eqc Type"].count().unstack(fill_value=0)
)
totals.columns = pd.MultiIndex.from_product([["Total"], totals.columns.astype(str)])
summary = pd.concat([summary, totals], axis=1)

summary.index.name = "EQC Checklist"

# --- Add Status summary at the end ---
if "Status" in df.columns:
    status_summary = df.groupby(["Eqc Type", "Status"])["Eqc Type"].count().unstack(fill_value=0)
    if not status_summary.empty:
        status_summary.columns = pd.MultiIndex.from_product([["Status"], status_summary.columns.astype(str)])
        summary = pd.concat([summary, status_summary], axis=1)

# --- Alerts ---
alerts = []
today = pd.Timestamp.today()

# 1. REDO > 3 days
redo_col = df.get("EQC Stage Status", pd.Series(dtype=str)).astype(str).str.upper()
redo_df = df[redo_col == "REDO"]
if not redo_df.empty:
    max_dates = redo_df.groupby(["Eqc Type", "Location L1"])["Date"].max()
    for (eqc, bldg), last_date in max_dates.items():
        if pd.notna(last_date) and (today - last_date).days > 3:
            alerts.append(f"ALERT: {eqc} in {bldg} is REDO for more than 3 days (last: {last_date.date()})")

# 2. IN_PROGRESS > 3 weeks without stage change
status_col = df.get("Status", pd.Series(dtype=str)).astype(str)
inprog_df = df[status_col == "IN_PROGRESS"]
if not inprog_df.empty:
    min_dates = inprog_df.groupby(["Eqc Type", "Location L1", "Stage_Norm"])["Date"].min()
    for (eqc, bldg, stage), first_date in min_dates.items():
        if pd.notna(first_date) and (today - first_date).days > 21:
            alerts.append(f"ALERT: {eqc} in {bldg} stuck in {stage} for more than 3 weeks (since: {first_date.date()})")

# --- Save outputs ---
summary_reset = summary.reset_index()
summary_reset.to_csv("EQC_Summary.csv", index=False)
pd.DataFrame(alerts, columns=["Alerts"]).to_csv("EQC_Alerts.csv", index=False)

print("✅ Summary saved as EQC_Summary.csv")
print("✅ Alerts saved as EQC_Alerts.csv")
print("Raw entries:", len(df))
print("Summary total count:", summary.select_dtypes(include=["number"]).sum().sum())

# Calculate totals per building
# Ensure columns are tuples
summary.columns = pd.MultiIndex.from_tuples([c if isinstance(c, tuple) else ("", str(c)) for c in summary.columns])
building_totals = {}
for bldg in sorted({c[0] for c in summary.columns if c[0] and c[0] not in ("Total", "Status")}):
    cols = [c for c in summary.columns if c[0] == bldg and c[1] != "Total"]
    if cols:
        building_totals[(bldg, "Total")] = summary.loc[:, cols].sum(axis=1)

if building_totals:
    bt_df = pd.concat(building_totals, axis=1)
    summary = pd.concat([summary, bt_df], axis=1)

# Calculate grand total row (numeric only)
numeric_sum = summary.select_dtypes(include=["number"]).sum()
if not numeric_sum.empty:
    grand_total = numeric_sum.to_frame().T
    grand_total.index = ["TOTAL"]
    summary_final = pd.concat([summary, grand_total], axis=0, sort=False)
else:
    summary_final = summary.copy()

# Sort columns: building, stage order, Totals at end
stage_rank = {s: i for i, s in enumerate(stage_order)}
def sort_key(col: Tuple[str, str]):
    building, stage = col
    if building == "Total":
        return (2, "ZZZ", stage_rank.get(stage, 99), str(stage))
    if building == "":
        return (3, "ZZZ", 99, str(stage))
    return (1, building, stage_rank.get(stage, 99), str(stage))

# If there are duplicate MultiIndex columns, combine them by summing (avoids non-unique MultiIndex)
if isinstance(summary_final.columns, pd.MultiIndex) and not summary_final.columns.is_unique:
    summary_final = summary_final.groupby(level=list(range(summary_final.columns.nlevels)), axis=1).sum()

sorted_cols = sorted(summary_final.columns.tolist(), key=sort_key)
summary_final = summary_final.reindex(columns=sorted_cols)

# --- Save outputs ---
summary_final_reset = summary_final.reset_index()
summary_final_reset.to_csv("EQC_Summary_Combined_WithTotals.csv", index=False)

print("✅ Summary (with building and overall totals) saved as EQC_Summary_Combined_WithTotals.csv")
summary_final_reset.to_csv("EQC_Summary_Combined_WithTotals.csv", index=False)

# --- Also write EQC_Summary_WithTotals.csv in the flattened tuple-string header format (reference style) ---
def flatten_col(col):
    # col may be tuple (building, stage) or a plain string
    if isinstance(col, tuple):
        return f"('{col[0]}', '{col[1]}')"
    return str(col)

# Work with a copy
out_df = summary_final.copy()

# Reset index to get 'EQC Checklist' as a column
out_df_reset = out_df.reset_index()

# Flatten column names
flat_cols = []
for col in out_df_reset.columns:
    if col == out_df_reset.columns[0]:
        # the index column name (likely 'EQC Checklist') keep as-is
        flat_cols.append(str(col))
    else:
        flat_cols.append(flatten_col(col))

out_df_reset.columns = flat_cols

# Ensure IN_PROGRESS and PASSED columns exist (they may come from ('Status','IN_PROGRESS'))
status_in_prog = None
status_passed = None
for c in out_df_reset.columns:
    if "('Status', 'IN_PROGRESS')" == c:
        status_in_prog = c
    if "('Status', 'PASSED')" == c or "('Status', 'PASS')" == c:
        status_passed = c

# Create clean IN_PROGRESS / PASSED columns (unprefixed) if present
if status_in_prog:
    out_df_reset['IN_PROGRESS'] = out_df_reset[status_in_prog]
if status_passed:
    out_df_reset['PASSED'] = out_df_reset[status_passed]

# Compute Total Count as sum across columns where header starts with "('Total',"
total_cols = [c for c in out_df_reset.columns if c.startswith("('Total',")]
if total_cols:
    out_df_reset['Total Count'] = out_df_reset[total_cols].sum(axis=1)
else:
    # fallback: sum numeric columns
    out_df_reset['Total Count'] = out_df_reset.select_dtypes(include=['number']).sum(axis=1)

# Reorder columns: make the EQC Checklist (index column) the leftmost, then building-stage columns,
# then IN_PROGRESS / PASSED (if present), and finally Total Count.
eqc_col = str(out_df_reset.columns[0])
building_stage_cols = [c for c in out_df_reset.columns if isinstance(c, str) and c.startswith("('")]

ordered = [eqc_col] + building_stage_cols
if 'IN_PROGRESS' in out_df_reset.columns:
    ordered += ['IN_PROGRESS']
if 'PASSED' in out_df_reset.columns:
    ordered += ['PASSED']
ordered += ['Total Count']

# Ensure uniqueness while preserving order and only keep columns that exist
seen = set()
final_cols = [c for c in ordered if c in out_df_reset.columns and (c not in seen and not seen.add(c))]

out_df_final = out_df_reset[final_cols]

# Append stage-total rows (one row per stage with EQC Checklist set to stage name and Total Count set to total)
grand = None
if 'TOTAL' in summary_final.index:
    grand = summary_final.loc['TOTAL']
else:
    grand = summary_final.sum()

stage_rows = []
for stage in stage_order:
    # sum across all columns where second element == stage
    s = 0
    for col in summary_final.columns:
        try:
            if isinstance(col, tuple) and str(col[1]) == str(stage):
                s += grand.get(col, 0)
        except Exception:
            continue
    row = {c: '' for c in out_df_final.columns}
    # put stage name in EQC Checklist column and value in Total Count
    row[eqc_col] = stage
    row['Total Count'] = s
    stage_rows.append(row)

if stage_rows:
    out_df_final = pd.concat([out_df_final, pd.DataFrame(stage_rows)], ignore_index=True, sort=False)

# Save
out_df_final.to_csv('EQC_Summary_WithTotals.csv', index=False)
print("✅ EQC_Summary_WithTotals.csv written in reference format")

# --- Also create a tidy by-building CSV where 'Building' is a proper column and
# similar contractor-specific checklist names are collapsed into a base name
# (split on the first '.') and summed per building. Back up existing file first.
try:
    src_path = 'EQC_Summary_WithTotals.csv'
    bak_path = 'EQC_Summary_WithTotals.csv.bak'
    # back up user-edited file if present
    if os.path.exists(src_path):
        # If a previous backup exists, overwrite it
        os.replace(src_path, bak_path)

    # Build a long-form DataFrame from summary_final
    # collect building names from raw data; treat empty Location L1 as 'UNKNOWN' so no rows are dropped
    raw_buildings = list(df.get('Location L1', pd.Series(dtype=str)).unique())
    buildings = sorted({(b if b else 'UNKNOWN') for b in raw_buildings})

    rows = []
    for bldg in buildings:
        # gather columns for this building for each stage
        cols = [(b, s) for (b, s) in summary_final.columns if b == bldg and s in stage_order]
        # ensure we have all stages
        for eqc in summary_final.index:
            if str(eqc).upper() == 'TOTAL':
                continue
            row = {"EQC Checklist": eqc, "Building": bldg}
            for s in stage_order:
                val = 0
                # lookup column: in summary_final the empty building may be stored as '' so map 'UNKNOWN' -> ''
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
                else:
                    val = 0
                row[s] = val
            row['Total Count'] = sum(row[s] for s in stage_order)
            rows.append(row)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print('No per-building rows to write.')
    else:
        # Collapse contractor-specific names into a base name (split on first '.')
        # smarter base-name extraction: prefer text before '.' or ':'; otherwise
        # group names sharing first-2 tokens and return their longest common token prefix
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
            
            # Handle "Checklist for ..." names - preserve full name
            if s_norm.startswith('checklist for '):
                # Remove contractor suffix if present (double space + name pattern)
                cleaned = re.sub(r'\s{2,}[A-Za-z]+\s*$', '', s)
                cleaned = re.sub(r'\s+\([^)]+\)\s*$', '', cleaned)  # Remove (Contractor Name)
                return cleaned.strip()
            
            # if dot or colon present, prefer text before it (common vendor separator)
            if '.' in s:
                return s.split('.', 1)[0].strip()
            if ':' in s:
                return s.split(':', 1)[0].strip()
            # fallback: find other names that share first two tokens
            tokens = s.split()
            if len(tokens) >= 2:
                key = ' '.join(tokens[:2])
                same_prefix = [x for x in summary_final.index.astype(str) if x.startswith(key)]
                if len(same_prefix) > 1:
                    lcp = longest_common_prefix_tokens(same_prefix)
                    return lcp if lcp else key
            # if no grouping, return up to first 4 tokens to avoid very long names
            return ' '.join(tokens[:4]).strip()

        long_df['BaseName'] = long_df['EQC Checklist'].apply(base_name)

        # Aggregate by BaseName + Building
        agg = long_df.groupby(['BaseName', 'Building'])[stage_order].sum().reset_index()
        # Ensure numeric and fill missing
        for s in stage_order:
            agg[s] = pd.to_numeric(agg[s], errors='coerce').fillna(0).astype(int)

        # Apply cumulative logic: Pre = Pre + During + Post; During = During + Post
        if all(col in agg.columns for col in ['Pre', 'During', 'Post']):
            agg['Pre'] = agg['Pre'] + agg['During'] + agg['Post']
            agg['During'] = agg['During'] + agg['Post']

        # Compute Total Count after cumulative transformation
        agg['Total Count'] = agg[stage_order].sum(axis=1)

        # Also add an overall BaseName total (across buildings)
        overall = agg.groupby('BaseName')[stage_order].sum().reset_index()
        overall['Building'] = 'ALL'
        overall['Total Count'] = overall[stage_order].sum(axis=1)

        final_by_building = pd.concat([agg, overall], ignore_index=True, sort=False)

        # Reorder columns
        cols_order = ['BaseName', 'Building'] + stage_order + ['Total Count']
        final_by_building = final_by_building[cols_order]

        # Save to CSV (this will replace the reference-style file; backup saved above)
        final_by_building.to_csv(src_path, index=False)
        print(f"✅ Wrote tidy per-building summary with collapsed names to {src_path} (backup -> {bak_path})")
        # Also produce a wide-format CSV with buildings as top-level columns and stages as sub-columns
        try:
            # agg has columns: BaseName, Building, Pre, During, ...
            wide_idx = agg.set_index(['BaseName', 'Building'])[stage_order]
            # unstack buildings -> columns (stage, building)
            # use final_by_building (which includes the 'ALL' aggregated row) to avoid missing totals
            wide_idx = final_by_building.set_index(['BaseName', 'Building'])[stage_order]
            wide_un = wide_idx.unstack(level='Building', fill_value=0)
            # swap to (building, stage)
            wide_un.columns = wide_un.columns.swaplevel(0, 1)

            # Add per-building 'Total' sub-column (sum of stages) for every building present
            buildings_set = sorted({c[0] for c in wide_un.columns})
            for b in buildings_set:
                # compute sum across available stages for this building
                stage_cols = [(b, s) for s in stage_order if (b, s) in wide_un.columns]
                if stage_cols:
                    wide_un[(b, 'Total')] = wide_un[stage_cols].sum(axis=1)

            # reorder buildings alphabetically but keep 'ALL' at the end if present
            buildings = sorted({c[0] for c in wide_un.columns})
            if 'ALL' in buildings:
                buildings.remove('ALL')
                buildings.append('ALL')

            # reorder columns by buildings and include Total as last stage per building
            ordered_cols = []
            for b in buildings:
                for s in stage_order + ['Total']:
                    if (b, s) in wide_un.columns:
                        ordered_cols.append((b, s))

            wide_un = wide_un.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))

            # Append grand TOTAL row (sum across BaseName rows) so no data is missing
            wide_un.loc['TOTAL'] = wide_un.sum(axis=0)

            # Reset index so BaseName is the leftmost column and write CSV (MultiIndex header preserved)
            wide_un_reset = wide_un.reset_index()
            wide_un_reset.to_csv('EQC_Summary_ByBuilding_Wide.csv', index=False)
            print('✅ Wrote wide-format per-building summary to EQC_Summary_ByBuilding_Wide.csv')
        except Exception as e:
            print('Failed to write wide-format per-building summary:', str(e))
except Exception as e:
    print('Failed to write per-building collapsed summary:', str(e))
