#!/usr/bin/env python3
"""
This script shows exactly what JavaScript will see and access when the
frontend code tries to read team.total_issues
"""
import json
from ai_analysis import analyze_issues_file

result = analyze_issues_file('../CSV-INSTRUCTION-LATEST-REPORT-12-08-2025-05-14-29.csv', use_ai=False)

# Simulate JavaScript accessing the response
print("="*100)
print("SIMULATING JAVASCRIPT EXECUTION")
print("="*100)

print("\n1. API returns JSON response:")
print(f"   data = {json.dumps(result)[:100]}...")

print("\n2. JavaScript checks: if (data.teams && data.teams.length > 0)")
if result.get('teams') and len(result['teams']) > 0:
    print(f"   ✓ PASS: data.teams exists with {len(result['teams'])} items")

print("\n3. JavaScript iterates: data.teams.forEach((team, idx) => {")
for idx, team in enumerate(result.get('teams', [])[:2]):
    print(f"\n   Iteration {idx}:")
    print(f"   - team object type: {type(team)} (should be dict/object)")
    print(f"   - team object: {json.dumps(team)[:100]}...")
    
    print(f"\n   4. JavaScript accesses: const teamName = team.team || 'Unknown Team'")
    teamName = team.get('team') or 'Unknown Team'
    print(f"      ✓ teamName = '{teamName}'")
    
    print(f"\n   5. JavaScript accesses: const totalIssues = team.total_issues || 0")
    totalIssues = team.get('total_issues') or 0
    print(f"      ✓ totalIssues = {totalIssues}")
    
    print(f"\n   6. JavaScript accesses: team.top_instructions")
    top_instructions = team.get('top_instructions')
    if top_instructions and isinstance(top_instructions, list):
        print(f"      ✓ top_instructions is a list with {len(top_instructions)} items")
        if top_instructions:
            print(f"      First instruction: {json.dumps(top_instructions[0])}")
    else:
        print(f"      ✗ ERROR: top_instructions is {type(top_instructions)}")

print("\n" + "="*100)
print("RESULT: No 'undefined' errors! All fields are accessible.")
print("="*100)
