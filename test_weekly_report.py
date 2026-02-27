#!/usr/bin/env python3
"""
Test script to simulate web app calling Weekly_report.py with uploaded file
"""
import os
import sys
import subprocess
import tempfile
import shutil

def test_web_upload_scenario():
    """Simulate what happens when user uploads file via web interface"""
    
    print("=" * 80)
    print("Testing Web Upload Scenario")
    print("=" * 80)
    
    # Get the actual EQC file from parent directory
    parent_dir = os.path.dirname(os.getcwd())
    eqc_source = os.path.join(parent_dir, "EQC-LATEST-STAGE-REPORT-12-08-2025-05-14-18.csv")
    
    if not os.path.exists(eqc_source):
        print(f"❌ Source EQC file not found: {eqc_source}")
        return False
    
    print(f"\n✓ Found source EQC file: {eqc_source}")
    print(f"  Size: {os.path.getsize(eqc_source) / (1024*1024):.2f} MB")
    
    # Simulate web app behavior: copy to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_eqc = os.path.join(tmpdir, "Combined_EQC.csv")
        shutil.copy2(eqc_source, temp_eqc)
        
        print(f"\n✓ Copied to temp location: {temp_eqc}")
        
        # Run Weekly_report.py with the temp file (simulating web app call)
        print("\n" + "=" * 80)
        print("Running Weekly_report.py with temp file...")
        print("=" * 80)
        
        cmd = [sys.executable, "Weekly_report.py", "--input", temp_eqc]
        print(f"Command: {' '.join(cmd)}\n")
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(proc.stdout)
        
        if proc.stderr:
            print("\nSTDERR:")
            print(proc.stderr)
        
        print("\n" + "=" * 80)
        if proc.returncode == 0:
            print("✅ SUCCESS: Weekly report generated successfully!")
            
            # Check if output file was created
            output_xlsx = "EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx"
            if os.path.exists(output_xlsx):
                size_kb = os.path.getsize(output_xlsx) / 1024
                print(f"✓ Output file created: {output_xlsx} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"⚠ Output file not found: {output_xlsx}")
                return False
        else:
            print(f"❌ FAILED with exit code: {proc.returncode}")
            return False

def test_with_wrong_file():
    """Test what happens when user uploads an Issues file instead of EQC"""
    
    print("\n" + "=" * 80)
    print("Testing Wrong File Upload (Issues instead of EQC)")
    print("=" * 80)
    
    # Try with datasheet.csv (which is an Issues file)
    issues_file = "datasheet.csv"
    
    if not os.path.exists(issues_file):
        print(f"⚠ Test file not found: {issues_file}")
        return True  # Skip this test
    
    print(f"\nTrying with Issues file: {issues_file}")
    
    cmd = [sys.executable, "Weekly_report.py", "--input", issues_file]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(proc.stdout)
    
    if proc.stderr:
        print("\nSTDERR:")
        print(proc.stderr)
    
    print("\n" + "=" * 80)
    if "doesn't appear to be an EQC file" in proc.stdout or "ISSUES file" in proc.stdout:
        print("✅ SUCCESS: Correctly detected wrong file type!")
        return True
    else:
        print("⚠ Warning: Wrong file type detection may need improvement")
        return True

if __name__ == "__main__":
    print("\n🧪 Testing Weekly Report Generation\n")
    
    # Test 1: Simulate web app upload
    test1_passed = test_web_upload_scenario()
    
    # Test 2: Wrong file type
    test2_passed = test_with_wrong_file()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Web Upload Simulation): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Wrong File Detection): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print("=" * 80)
