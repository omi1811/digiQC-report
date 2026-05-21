#!/usr/bin/env python3
"""
Test script to demonstrate the corrected building name normalization.
This shows how the approach:
1. Learns valid building names from NON-EMPTY Location L1 values
2. Fills EMPTY Location L1 by extracting building names from EQC column
3. Only uses building names that already exist in Location L1
4. Never modifies Location L1 values that are already filled
"""

import pandas as pd
import building_normalizer as bn

# Load the EQC CSV
csv_file = "../EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv"

print("=" * 80)
print("TESTING FILL EMPTY LOCATION L1 - CORRECTED APPROACH")
print("=" * 80)

try:
    # Try reading with tab separator first
    df = pd.read_csv(csv_file, sep='\t', dtype=str, keep_default_na=False)
    print(f"\n✓ Loaded CSV file: {csv_file}")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    exit(1)

print(f"  Total rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()[:5]}... (showing first 5)")

# Show the current state
print("\n" + "=" * 80)
print("BEFORE NORMALIZATION")
print("=" * 80)

if 'Location L1' in df.columns:
    unique_l1 = df['Location L1'].unique()
    print(f"\nUnique values in Location L1 (CANONICAL NAMES): {len(unique_l1)}")
    for val in sorted([str(v).strip() for v in unique_l1 if pd.notna(v)]):
        count = (df['Location L1'] == val).sum()
        print(f"  - {val}: {count} rows")
else:
    print("✗ Location L1 column not found!")
    exit(1)

if 'Building' in df.columns:
    unique_building = df['Building'].unique()
    print(f"\nUnique values in Building column: {len(unique_building)}")
    unique_list = sorted([str(v).strip() for v in unique_building if pd.notna(v)])
    if len(unique_list) <= 15:
        for val in unique_list:
            count = (df['Building'] == val).sum()
            print(f"  - {val}: {count} rows")
    else:
        print(f"  (Too many values to display, showing first 10 and last 5)")
        for val in unique_list[:10]:
            count = (df['Building'] == val).sum()
            print(f"  - {val}: {count} rows")
        print(f"  ... ({len(unique_list) - 15} more) ...")
        for val in unique_list[-5:]:
            count = (df['Building'] == val).sum()
            print(f"  - {val}: {count} rows")

# Extract valid building names (from non-empty Location L1 values)
valid_names = bn.extract_valid_building_names(df)
print(f"\n✓ Learned VALID/CANONICAL building names from NON-EMPTY Location L1:")
for name in sorted(valid_names):
    print(f"  - {name}")

# Apply filling
print("\n" + "=" * 80)
print("APPLYING FILL EMPTY LOCATION L1")
print("=" * 80)
print("\nFilling logic:")
print("  1. Learn valid building names from non-empty Location L1 values")
print("  2. For rows with EMPTY Location L1:")
print("     - Extract building name from EQC column")
print("     - If extracted name exists in valid names → fill Location L1")
print("     - Otherwise → fill with 'Unknown'")
print("  3. NEVER modify Location L1 values that are already filled")

df_normalized = bn.fill_empty_location_l1(df)
print("\n✓ Filling complete!")

# Show the normalized state
print("\n" + "=" * 80)
print("AFTER NORMALIZATION")
print("=" * 80)

if 'Location L1' in df_normalized.columns:
    unique_l1_norm = df_normalized['Location L1'].unique()
    print(f"\nUnique values in Location L1 (AFTER normalization): {len(unique_l1_norm)}")
    for val in sorted([str(v).strip() for v in unique_l1_norm if pd.notna(v)]):
        count = (df_normalized['Location L1'] == val).sum()
        print(f"  - {val}: {count} rows")

if 'Building' in df_normalized.columns:
    unique_building_norm = df_normalized['Building'].unique()
    print(f"\nUnique values in Building column (AFTER normalization): {len(unique_building_norm)}")
    for val in sorted([str(v).strip() for v in unique_building_norm if pd.notna(v)]):
        count = (df_normalized['Building'] == val).sum()
        print(f"  - {val}: {count} rows")

# Get project summary
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)
summary = bn.get_project_building_summary(df_normalized)
print("\nBuilding distribution in this project (after normalization):")
for building, count in summary.items():
    print(f"  {building}: {count}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nKey behavior:")
print("✓ Location L1 values that were already filled → UNCHANGED")
print("✓ Empty Location L1 filled with building names from EQC column")
print("✓ Only building names that exist in non-empty Location L1 are used")
print("✓ Unmatched extracted names → 'Unknown'")
print("✓ This approach is project-specific (uses actual Location L1 names)")
