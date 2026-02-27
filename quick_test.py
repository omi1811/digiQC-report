import subprocess, sys

# Quick test to see debug output
cmd = [sys.executable, "Weekly_report.py", "--input", r"C:\Users\shrot\OneDrive\Desktop\digiqc-report\EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv"]
proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

print("=== STDOUT ===")
print(proc.stdout)

if proc.stderr:
    print("\n=== STDERR ===")
    print(proc.stderr)

print(f"\n=== EXIT CODE: {proc.returncode} ===")
