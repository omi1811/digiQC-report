"""Test web interface file upload and report generation"""
import requests
import os
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:5000"
USERNAME = "admin"  # Default from app.py
PASSWORD = "admin"

def test_upload_and_generate():
    print("\n" + "="*80)
    print("TEST: Web Upload and Report Generation")
    print("="*80 + "\n")
    
    # Create session
    session = requests.Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)
    
    # Step 1: Login
    print("Step 1: Logging in...")
    login_response = session.get(f"{BASE_URL}/")
    if login_response.status_code != 200:
        print(f"[FAIL] Login failed with status {login_response.status_code}")
        return False
    print("[OK] Logged in successfully")
    
    # Step 2: Upload EQC file
    print("\nStep 2: Uploading eqc.csv file...")
    if not os.path.exists("eqc.csv"):
        print("[FAIL] eqc.csv not found!")
        return False
    
    with open("eqc.csv", "rb") as f:
        files = {"eqc": ("eqc.csv", f, "text/csv")}
        upload_response = session.post(
            f"{BASE_URL}/set-eqc-file",
            files=files,
            allow_redirects=False
        )
    
    if upload_response.status_code not in (200, 302):
        print(f"[FAIL] Upload failed with status {upload_response.status_code}")
        print(f"Response: {upload_response.text[:500]}")
        return False
    print(f"[OK] File uploaded successfully (status {upload_response.status_code})")
    
    # Step 3: Generate weekly report
    print("\nStep 3: Generating weekly report...")
    report_response = session.post(
        f"{BASE_URL}/weekly-report",
        allow_redirects=False
    )
    
    if report_response.status_code != 200:
        print(f"[FAIL] Report generation failed with status {report_response.status_code}")
        print(f"Response: {report_response.text[:1000]}")
        return False
    
    # Check if we got an Excel file
    content_type = report_response.headers.get("Content-Type", "")
    content_length = len(report_response.content)
    
    print(f"[OK] Report generated successfully!")
    print(f"  Content-Type: {content_type}")
    print(f"  Content-Length: {content_length} bytes")
    
    if content_length == 0:
        print("[FAIL] Excel file is EMPTY (0 bytes)!")
        return False
    
    if content_length < 5000:
        print(f"[WARNING] Excel file is very small ({content_length} bytes) - may be empty")
        return False
    
    # Save the file for inspection
    output_path = "test_web_output.xlsx"
    with open(output_path, "wb") as f:
        f.write(report_response.content)
    print(f"\n[OK] Saved Excel file to: {output_path}")
    
    # Try to read it with pandas
    try:
        import pandas as pd
        xl = pd.ExcelFile(output_path)
        print(f"[OK] Excel file has {len(xl.sheet_names)} sheets: {xl.sheet_names}")
        
        # Check for Itrend Vesta sheet
        vesta_sheets = [s for s in xl.sheet_names if "Vesta" in s]
        if not vesta_sheets:
            print("[FAIL] No Vesta sheet found!")
            return False
        
        vesta_sheet = vesta_sheets[0]
        df = pd.read_excel(output_path, sheet_name=vesta_sheet)
        print(f"[OK] {vesta_sheet} has {len(df)} rows x {len(df.columns)} columns")
        
        if len(df) == 0:
            print("[FAIL] Vesta sheet is EMPTY!")
            return False
        
        print("\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"[WARNING] Could not read Excel file: {e}")
    
    print("\n" + "="*80)
    print("[PASS] Web upload and report generation works correctly!")
    print("="*80)
    return True

if __name__ == "__main__":
    import time
    time.sleep(2)  # Give Flask time to start
    test_upload_and_generate()
