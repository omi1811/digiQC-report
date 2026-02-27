#!/usr/bin/env python3
"""
Simulate the exact API response that will be returned to the frontend.
This shows the JSON structure that the JavaScript code will receive.
"""
import json
import sys
from ai_analysis import analyze_issues_file

# Simulate loading the issues file from session
result = analyze_issues_file('../CSV-INSTRUCTION-LATEST-REPORT-12-08-2025-05-14-29.csv', use_ai=False)

print("="*100)
print("API RESPONSE (Exact JSON that will be sent to frontend)")
print("="*100)

# Print as JSON (exactly as Flask would return it)
json_response = json.dumps(result, indent=2)
print(json_response)

print("\n" + "="*100)
print("VALIDATION")
print("="*100)

# Validate response structure
if 'error' in result:
    print(f"✗ Error response: {result['error']}")
    sys.exit(1)

if 'teams' not in result:
    print("✗ Missing 'teams' key in response!")
    sys.exit(1)

teams = result.get('teams', [])
print(f"✓ Response has 'teams' key with {len(teams)} teams")

for idx, team in enumerate(teams):
    if not isinstance(team, dict):
        print(f"✗ Team {idx} is not a dict: {type(team)}")
        sys.exit(1)
    
    if 'team' not in team:
        print(f"✗ Team {idx} missing 'team' key")
        sys.exit(1)
    
    if 'total_issues' not in team:
        print(f"✗ Team {idx} missing 'total_issues' key")
        sys.exit(1)
    
    if 'top_instructions' not in team:
        print(f"✗ Team {idx} missing 'top_instructions' key")
        sys.exit(1)
    
    if not isinstance(team['top_instructions'], list):
        print(f"✗ Team {idx} 'top_instructions' is not a list")
        sys.exit(1)

print(f"✓ All {len(teams)} teams have correct structure")
print(f"✓ Note: {result.get('note', '')}")

print("\n" + "="*100)
print("FRONTEND SIMULATION - What the user will see:")
print("="*100)

for team in teams[:3]:
    print(f"\n▼ {team['team']} [{team['total_issues']} issues]")
    for inst in team['top_instructions'][:3]:
        print(f"   {inst['description'][:60]:<60} {inst['count']}")

print(f"\n... and {len(teams) - 3} more teams")
