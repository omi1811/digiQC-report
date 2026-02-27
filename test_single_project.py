#!/usr/bin/env python3
"""
Test script to verify single-project file upload works correctly
"""
import os
import sys
import subprocess
import tempfile
import pandas as pd

def create_single_project_file():
    """Create a test EQC file with only one project's data"""
    
    # Read the full EQC file
    parent_dir = os.path.dirname(os.getcwd())
    full_eqc = os.path.join(parent_dir, "EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv")
    
    if not os.path.exists(full_eqc):
        print(f"Source file not found: {full_eqc}")
        return None
    
    # Read and filter for one project only
    df = pd.read_csv(full_eqc, dtype=str, keep_default_na=False)
    print(f"\nFull file has {len(df)} rows")
    print(f"Projects in full file: {df['Project'].unique().tolist()}")
    
    # Filter for just Itrend Futura project
    df_single = df[df['Project'].str.contains('Itrend Futura', case=False, na=False)].copy()
    print(f"\nFiltered to Itrend Futura only: {len(df_single)} rows")
    
    # Save to temp file
    temp_file = os.path.join(tempfile.gettempdir(), "test_single_project_eqc.csv")
    df_single.to_csv(temp_file, index=False)
    print(f"Created single-project file: {temp_file}")
    print(f"File size: {os.path.getsize(temp_file) / 1024:.1f} KB")
    
    return temp_file

def test_single_project_upload():
    """Test that single-project upload generates report with ONLY that project"""
    
    print("=" * 80)
    print("TEST: Single Project Upload")
    print("=" * 80)
    
    # Create test file with only one project
    single_project_file = create_single_project_file()
    
    if not single_project_file:
        print("Failed to create test file")
        return False
    
    print("\n" + "=" * 80)
    print("Running Weekly_report.py with single-project file...")
    print("=" * 80)
    
    # Run Weekly_report.py with explicit file
    cmd = [sys.executable, "Weekly_report.py", "--input", single_project_file]
    print(f"Command: {' '.join(cmd)}\n")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(proc.stdout)
    
    if proc.stderr:
        print("\nSTDERR:")
        print(proc.stderr)
    
    print("\n" + "=" * 80)
    
    if proc.returncode != 0:
        print(f"FAILED with exit code: {proc.returncode}")
        return False
    
    # Check the output
    output_xlsx = "EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx"
    if not os.path.exists(output_xlsx):
        print(f"Output file not found: {output_xlsx}")
        return False
    
    # Verify it only has LANDMARC project
    import openpyxl
    wb = openpyxl.load_workbook(output_xlsx)
    sheet_names = wb.sheetnames
    print(f"\nGenerated workbook has {len(sheet_names)} sheets:")
    for name in sheet_names:
        print(f"  - {name}")
    
    # Check if ONLY Futura sheets exist (no other projects)
    futura_sheets = [s for s in sheet_names if 'Futura' in s or 'FUTURA' in s]
    other_project_sheets = [s for s in sheet_names if ('Palacio' in s or 'Vesta' in s or 'City Life' in s or 'Landmarc' in s) and 'Futura' not in s]
    
    print(f"\nFutura sheets: {len(futura_sheets)}")
    print(f"Other project sheets: {len(other_project_sheets)}")
    
    if other_project_sheets:
        print("\n[FAIL] Found sheets for other projects! This means it read the wrong file!")
        print(f"Other projects found: {other_project_sheets}")
        return False
    
    if futura_sheets:
        print("\n[PASS] Only Futura project found in output - uploaded file was used correctly!")
        return True
    
    print("\n[WARN] No project sheets found at all")
    return False

if __name__ == "__main__":
    print("\nTesting Single-Project File Upload\n")
    
    success = test_single_project_upload()
    
    print("\n" + "=" * 80)
    if success:
        print("TEST PASSED: Uploaded single-project file was used correctly!")
    else:
        print("TEST FAILED: System is not using the uploaded file!")
    print("=" * 80)
