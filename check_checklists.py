#!/usr/bin/env python3
import pandas as pd
import glob
import os

# Get the combined EQC file used by the app
csv_files = []
if os.path.exists("Combined_EQC.csv"):
    csv_files = ["Combined_EQC.csv"]
else:
    csv_files = glob.glob("Itrend_*-Cumulative-digiQC-report_EQC_*.csv")

if csv_files:
    print(f"Processing: {csv_files[0]}")
    df = pd.read_csv(csv_files[0], dtype=str, keep_default_na=False)
    
    print(f"\nFile: {csv_files[0]}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Try to find checklist column
    checklist_col = None
    for col in df.columns:
        if 'checklist' in col.lower():
            checklist_col = col
            break
    
    if not checklist_col:
        print("\nNo Checklist column found. First 5 columns and their sample values:")
        for col in list(df.columns)[:5]:
            print(f"  {col}: {df[col].iloc[0]}")
    else:
        checklists = df[checklist_col].drop_duplicates().sort_values().tolist()
        print(f"\nTotal unique {checklist_col}: {len(checklists)}")
        print(f"\n{checklist_col}s:")
        for i, cl in enumerate(checklists, 1):
            print(f"{i:2}. {cl}")
else:
    print("No Combined_EQC.csv or Itrend files found. Available CSV files:")
    for f in os.listdir("."):
        if f.endswith(".csv"):
            print(f"  - {f}")
