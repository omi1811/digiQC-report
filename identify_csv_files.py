#!/usr/bin/env python3
"""
Helper script to identify CSV file types in the workspace.
Helps distinguish between EQC files and Issues files.
"""
import os
import pandas as pd
from pathlib import Path

def identify_csv_type(file_path):
    """Identify whether a CSV is an EQC file or Issues file."""
    try:
        # Read first row to check columns
        df = pd.read_csv(file_path, nrows=1, dtype=str, keep_default_na=False)
        columns = set(df.columns)
        
        # EQC file indicators
        eqc_columns = {"Eqc Type", "Stage", "Inspector", "EQC"}
        # Issues file indicators  
        issues_columns = {"Reference ID", "Type L0", "Type L1", "Raised By", "Assigned Team"}
        
        eqc_matches = len(eqc_columns & columns)
        issues_matches = len(issues_columns & columns)
        
        if eqc_matches >= 2:
            return "EQC", df.columns.tolist()
        elif issues_matches >= 2:
            return "Issues", df.columns.tolist()
        else:
            return "Unknown", df.columns.tolist()
    except Exception as e:
        return f"Error: {str(e)}", []

def main():
    """Scan for CSV files and identify their types."""
    print("=" * 80)
    print("CSV File Type Identifier")
    print("=" * 80)
    
    # Check current directory
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    csv_files = []
    
    # Scan current directory
    for file in os.listdir(current_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(current_dir, file))
    
    # Scan parent directory
    if os.path.exists(parent_dir):
        for file in os.listdir(parent_dir):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(parent_dir, file))
    
    if not csv_files:
        print("\nNo CSV files found in current or parent directory.")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):\n")
    
    eqc_files = []
    issues_files = []
    unknown_files = []
    
    for csv_file in csv_files:
        file_type, columns = identify_csv_type(csv_file)
        rel_path = os.path.relpath(csv_file, current_dir)
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
        
        print(f"\n📄 {rel_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Type: {file_type}")
        
        if file_type == "EQC":
            eqc_files.append(csv_file)
            print(f"   ✓ This is an EQC file (use for Weekly_report.py)")
        elif file_type == "Issues":
            issues_files.append(csv_file)
            print(f"   ✓ This is an Issues file")
        else:
            unknown_files.append(csv_file)
        
        if columns:
            print(f"   Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  EQC Files: {len(eqc_files)}")
    print(f"  Issues Files: {len(issues_files)}")
    print(f"  Unknown Files: {len(unknown_files)}")
    print("=" * 80)
    
    if eqc_files:
        print("\n✓ To generate weekly reports, use:")
        for eqc_file in eqc_files:
            rel_path = os.path.relpath(eqc_file, current_dir)
            print(f"   python Weekly_report.py --input \"{rel_path}\"")
    else:
        print("\n⚠ No EQC files found! Weekly_report.py needs an EQC export CSV.")
        print("   EQC files should have columns: Eqc Type, Stage, Inspector, Status, Date")

if __name__ == "__main__":
    main()
