# CSV File Issue - Resolved ✅

## Problem Summary
The `Weekly_report.py` script was failing because:
1. It was looking for `Combined_EQC.csv` which didn't exist
2. The `datasheet.csv` file is an **Issues CSV** (not an EQC CSV)
3. The actual **EQC CSV** was in the parent directory with a different name

## Files Identified

### In Parent Directory (`c:\Users\shrot\OneDrive\Desktop\digiqc-report\`)
- `EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv` - **EQC Data** ✓
  - Has columns: Date, Project, Inspector, Eqc Type, Stage, Status, etc.
  - Size: 1.52 MB
  
- `CSV-INSTRUCTION-LATEST-REPORT-12-08-2025-05-14-29.csv` - **Issues Data**
  - Has columns: Reference ID, Type L0, Type L1, Raised By, etc.
  - Size: 1.35 MB

### In Current Directory (`digiQC-report/`)
- `datasheet.csv` - **Issues Data** (NOT EQC)
  - Has columns: Reference ID, Project Name, Type L0, Type L1, etc.
  - Size: 0.14 MB

## Solutions Implemented

### 1. Fixed Date Column Error (Line 87)
**Before:**
```python
df["Date"] = pd.to_datetime(df.get("Date", ""), dayfirst=True, errors="coerce").dt.normalize()
```

**Problem:** When "Date" column doesn't exist, `df.get()` returns empty string, causing `.dt.normalize()` to fail

**After:**
```python
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
else:
    df["Date"] = pd.NaT
```

### 2. Enhanced File Auto-Detection
The script now:
- Searches current and parent directories for EQC files
- Looks for patterns: `EQC-*.csv`, `eqc-*.csv`, `EQC_*.csv`
- Validates files by checking for EQC columns (Eqc Type, Stage)
- Provides helpful error messages listing available CSV files

### 3. Created Helper Tool
Created `identify_csv_files.py` to help identify which CSV files are which type:
```bash
python identify_csv_files.py
```

## How to Use

### Option 1: Auto-Detection (Recommended)
Simply run the script - it will automatically find the EQC file:
```bash
python Weekly_report.py
```

### Option 2: Specify File Explicitly
Provide the exact path to the EQC file:
```bash
python Weekly_report.py --input "..\EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv"
```

### Option 3: Identify Files First
Run the identifier to see what CSV files you have:
```bash
python identify_csv_files.py
```

## Output Generated ✓
- `EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx` (18.6 KB)
- Individual project CSV reports for:
  - Itrend Palacio (327 rows)
  - Itrend Futura (3196 rows)
  - Itrend City Life (876 rows)
  - Itrend Vesta
  - Landmarc
  - Saheel Landmarc

## Key Takeaways

1. **EQC vs Issues Files**: Your project has TWO types of CSV files:
   - **EQC files**: For quality checklist data (use with `Weekly_report.py`)
   - **Issues files**: For issue tracking data (use with other scripts)

2. **File Location**: EQC files are in the parent directory, not the working directory

3. **Auto-Detection**: The script now automatically finds and validates EQC files

4. **Helper Tool**: Use `identify_csv_files.py` anytime to identify your CSV files

## Future Usage

When you get new EQC exports:
1. Place them in the parent directory OR
2. Name them starting with "EQC-" for auto-detection OR
3. Use `--input` parameter to specify the exact path

The script will automatically find and use the correct file!
