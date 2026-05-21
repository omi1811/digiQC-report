#!/usr/bin/env python3
"""
Test script to verify Location L1 variant normalization.
Tests that variants like "Tower A", "B tower", "Wing A" are normalized to canonical forms.
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

import building_normalizer as bn

# Create test dataframe with variants
test_data = {
    'Date': ['2026-01-01'] * 10,
    'Project': ['Futura'] * 10,
    'EQC': [
        'FUTURA/Building A/Floor 1/Flat 101/101',
        'FUTURA/Building B/Floor 1/Flat 201/201',
        'FUTURA/Building A/Floor 2/Flat 102/102',
        'FUTURA/Building B/Floor 2/Flat 202/202',
        'FUTURA/Building A/Floor 3/Flat 103/103',
        'FUTURA/MLCP/Level 1/Pour-1/Pour-1',
        'FUTURA/Development/Section 1/Plot 1/Plot 1',
        'FUTURA/Unknown/Floor 5/Flat 501/501',
        'FUTURA/Building C/Floor 5/Flat 501/501',
        'FUTURA/Building C/Floor 6/Flat 601/601',
    ],
    'Location L1': [
        'Building A',      # Canonical form - appears most
        'Tower B',         # Variant - should normalize to Building B
        'Wing A',          # Variant - should normalize to Building A
        'Building B',      # Canonical form
        '',                # Empty - should be filled from EQC
        'MLCP',            # Already filled
        'Development',     # Already filled
        '',                # Empty - should become Unknown (can't extract)
        'B Tower',         # Variant - should normalize to Building B? or is it Building C from EQC?
        '',                # Empty - should be filled from EQC as Building C
    ],
    'Location L0': ['FUTURA'] * 10,
}

df = pd.DataFrame(test_data)

print("="*80)
print("TESTING LOCATION L1 VARIANT NORMALIZATION")
print("="*80)

print("\n### BEFORE NORMALIZATION ###\n")
print("Location L1 unique values and their counts:")
print(df['Location L1'].value_counts(dropna=False))
print(f"\nTotal rows: {len(df)}")

# Apply normalization (includes variant normalization + filling)
df_normalized = bn.fill_empty_location_l1(df)

print("\n### AFTER NORMALIZATION ###\n")
print("Location L1 unique values and their counts:")
print(df_normalized['Location L1'].value_counts(dropna=False))
print(f"\nTotal rows: {len(df_normalized)}")

print("\n### DETAILED RESULTS ###\n")
for idx, row in df_normalized.iterrows():
    print(f"Row {idx}: Location L1 = '{row['Location L1']}' | EQC = {row['EQC']}")

# Verify business rules
print("\n### VERIFICATION ###\n")

# Check that all rows have Location L1 filled
if df_normalized['Location L1'].isna().any() or (df_normalized['Location L1'] == '').any():
    print("❌ ERROR: Some Location L1 values are still empty!")
else:
    print("✓ All Location L1 values are filled (no empty values)")

# Check that we have canonical forms (Building A, Building B, etc.)
unique_l1 = set(df_normalized['Location L1'].unique())
print(f"\n✓ Unique Location L1 values after normalization: {sorted(unique_l1)}")

# Verify that variants were normalized
if 'Tower B' in df_normalized['Location L1'].values:
    print("❌ ERROR: 'Tower B' variant was not normalized!")
elif 'Wing A' in df_normalized['Location L1'].values:
    print("❌ ERROR: 'Wing A' variant was not normalized!")
else:
    print("✓ All variants successfully normalized to canonical forms")

print("\n### PROJECT SUMMARY ###\n")
summary = bn.get_project_building_summary(df_normalized)
for building, count in sorted(summary.items()):
    print(f"  {building}: {count}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
