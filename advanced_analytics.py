"""
Advanced Analytics Module for digiQC Report
Features:
1. Overdue Issues Dashboard
2. Visual Floor Plan Heatmap
3. Repeat Offenders Analysis (Location/Contractor)
4. Post-Approval Issue Detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
import re
from project_utils import canonicalize_project_name


class AdvancedAnalytics:
    """Advanced analytics for quality control and progress tracking."""
    
    def __init__(self):
        pass
    
    # ========== DEMO FILTERING ==========
    
    def _filter_demo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove DEMO/test projects."""
        demo_cols = ['Project', 'Project Name', 'Location L0']
        for col in demo_cols:
            if col in df.columns:
                mask = df[col].astype(str).str.contains('DEMO', case=False, na=False)
                df = df[~mask]
        return df
    
    # ========== OVERDUE ISSUES DASHBOARD ==========
    
    def analyze_overdue_issues(self, issues_df: pd.DataFrame) -> dict:
        """
        Analyze issues that have passed their deadline.
        
        Args:
            issues_df: DataFrame with Instructions data
        
        Returns:
            dict with overdue issues grouped by various dimensions
        """
        df = self._filter_demo(issues_df.copy())
        
        # Add project identification
        def get_project_name(row):
            for col in ['Project Name', 'Location L0', 'Project']:
                if col in row.index:
                    name = str(row[col]).strip()
                    if name and name not in ('', 'nan', 'None'):
                        return canonicalize_project_name(name)
            return 'Unknown'
        
        df['__Project'] = df.apply(get_project_name, axis=1)
        
        results = {
            'summary': {},
            'overdue_issues': [],
            'by_team': {},
            'by_type': {},
            'by_location': {},
            'by_project': {},
            'total_by_project': {},
            'raised_by_project': {},
            'project_summary': {}
        }
        
        # Parse deadline datetime
        if 'Deadline Date' not in df.columns:
            return {'error': 'Missing Deadline Date column'}
        
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
        print(f"[DEBUG] Sample 'Deadline Date' values: {df['Deadline Date'].head(10).tolist()}")
        
        def parse_deadline(row):
            """Parse deadline from Date + Time columns."""
            try:
                date_str = str(row.get('Deadline Date', '')).strip()
                time_str = str(row.get('Deadline Time', '')).strip()
                
                # Skip empty/null values
                if not date_str or date_str in ('', 'nan', 'None', 'NaT', 'NaN'):
                    return None
                
                # Parse date - try multiple formats
                parsed_dt = None
                for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y'):
                    try:
                        parsed_dt = datetime.strptime(date_str, fmt)
                        break
                    except:
                        continue
                
                if parsed_dt is None:
                    return None
                
                # Add time if available
                if time_str and time_str not in ('', 'nan', 'None', 'NaT', 'NaN'):
                    try:
                        # Try parsing time (HH:MM AM/PM or HH:MM am/pm)
                        time_str_lower = time_str.lower()
                        for time_fmt in ('%I:%M %p', '%I:%M%p', '%H:%M'):
                            try:
                                time_obj = datetime.strptime(time_str_lower, time_fmt)
                                parsed_dt = parsed_dt.replace(hour=time_obj.hour, minute=time_obj.minute)
                                break
                            except:
                                continue
                    except:
                        pass
                
                return parsed_dt
            except Exception as e:
                print(f"[DEBUG] Error parsing deadline: {e}")
                return None
        
        df['__Deadline'] = df.apply(parse_deadline, axis=1)
        df_with_deadline = df[df['__Deadline'].notna()].copy()
        
        # Debug info
        print(f"[DEBUG] Total issues in input: {len(df)}")
        print(f"[DEBUG] Issues with valid deadline: {len(df_with_deadline)}")
        
        now = datetime.now()
        print(f"[DEBUG] Current datetime: {now}")
        
        # Parse Current Status Updated On (Date) for closed issues
        def parse_status_date(date_str):
            """Parse status update date (DD/MM/YYYY)."""
            try:
                date_str = str(date_str).strip()
                if not date_str or date_str in ('', 'nan', 'None', 'NaT', 'NaN'):
                    return None
                for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'):
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
                return None
            except:
                return None
        
        # Parse status date if column exists
        if 'Current Status Updated On (Date)' in df_with_deadline.columns:
            df_with_deadline['__StatusDate'] = df_with_deadline['Current Status Updated On (Date)'].apply(parse_status_date)
        else:
            df_with_deadline['__StatusDate'] = None
        
        # Determine overdue based on status
        def is_overdue(row):
            status = str(row.get('Current Status', '')).strip().upper()
            deadline = row['__Deadline']
            
            # For CLOSED/RESPONDED: overdue if closed/responded AFTER deadline
            if status in ('CLOSED', 'RESPONDED'):
                status_date = row.get('__StatusDate')
                if status_date and pd.notna(status_date):
                    return status_date > deadline
                else:
                    # If no status date, assume not overdue (closed on time)
                    return False
            
            # For RAISED/REJECTED or other open statuses: overdue if deadline passed
            elif status in ('RAISED', 'REJECTED'):
                return now > deadline
            
            # For any other status, don't count as overdue
            else:
                return False
        
        df_with_deadline['__IsOverdue'] = df_with_deadline.apply(is_overdue, axis=1)
        df_with_deadline['__IsRaised'] = df_with_deadline['Current Status'].astype(str).str.upper().isin(['RAISED', 'REJECTED'])
        
        overdue_df = df_with_deadline[df_with_deadline['__IsOverdue']].copy()
        
        print(f"[DEBUG] Overdue issues found: {len(overdue_df)}")
        
        # Calculate days overdue
        def calc_days_overdue(row):
            status = str(row.get('Current Status', '')).strip().upper()
            deadline = row['__Deadline']
            
            if status in ('CLOSED', 'RESPONDED'):
                status_date = row.get('__StatusDate')
                if status_date and pd.notna(status_date):
                    return (status_date - deadline).days
                return 0
            else:
                return (now - deadline).days
        
        overdue_df['__DaysOverdue'] = overdue_df.apply(calc_days_overdue, axis=1)
        
        # Summary
        total_with_deadline = len(df_with_deadline)
        total_overdue = len(overdue_df)
        
        print(f"[DEBUG] Total with deadline: {total_with_deadline}, Total overdue: {total_overdue}")
        if total_overdue > 0:
            print(f"[DEBUG] Sample overdue days: {overdue_df['__DaysOverdue'].head(5).tolist()}")
        
        results['summary'] = {
            'total_issues_with_deadline': total_with_deadline,
            'total_overdue': total_overdue,
            'overdue_percentage': round(total_overdue / total_with_deadline * 100, 1) if total_with_deadline > 0 else 0
        }
        
        # Overdue issues list (sorted by days overdue)
        for _, row in overdue_df.sort_values('__DaysOverdue', ascending=False).head(50).iterrows():
            results['overdue_issues'].append({
                'project': str(row.get('__Project', 'Unknown')),
                'description': str(row.get('Description', ''))[:100],
                'type': str(row.get('Type L0', '')),
                'location': str(row.get('Location L0', '')),
                'assigned_team': str(row.get('Assigned Team', '')),
                'deadline': row['__Deadline'].strftime('%d/%m/%Y'),
                'days_overdue': int(row['__DaysOverdue']),
                'status': str(row.get('Current Status', ''))
            })
        
        # By Team
        if 'Assigned Team' in overdue_df.columns:
            team_counts = overdue_df['Assigned Team'].value_counts().head(10)
            for team, count in team_counts.items():
                results['by_team'][str(team)] = int(count)
        
        # By Type
        if 'Type L0' in overdue_df.columns:
            type_counts = overdue_df['Type L0'].value_counts().head(10)
            for typ, count in type_counts.items():
                results['by_type'][str(typ)] = int(count)
        
        # By Location
        if 'Location L0' in overdue_df.columns:
            loc_counts = overdue_df['Location L0'].value_counts().head(10)
            for loc, count in loc_counts.items():
                results['by_location'][str(loc)] = int(count)
        
        # By Project - with total, raised, and overdue counts
        if '__Project' in df_with_deadline.columns:
            # Total issues per project (with deadline)
            total_proj_counts = df_with_deadline['__Project'].value_counts()
            results['total_by_project'] = {proj: int(count) for proj, count in total_proj_counts.items()}
            
            # Raised issues per project
            raised_proj_counts = df_with_deadline[df_with_deadline['__IsRaised']].groupby('__Project').size()
            results['raised_by_project'] = {proj: int(count) for proj, count in raised_proj_counts.items()}
            
            # Overdue issues per project
            if len(overdue_df) > 0 and '__Project' in overdue_df.columns:
                proj_counts = overdue_df['__Project'].value_counts()
                results['by_project'] = {proj: int(count) for proj, count in proj_counts.items()}
            
            # Combined view: {project: {total: X, raised: Y, overdue: Z, percentage: W}}
            for proj in results['total_by_project'].keys():
                total = results['total_by_project'][proj]
                raised = results['raised_by_project'].get(proj, 0)
                overdue = results['by_project'].get(proj, 0)
                results['project_summary'][proj] = {
                    'total_issues': total,
                    'raised_issues': raised,
                    'overdue_issues': overdue,
                    'overdue_percentage': round(overdue / total * 100, 1) if total > 0 else 0
                }
        
        return results
    
    # ========== FLOOR HEATMAP ==========
    
    def analyze_floor_completion(self, eqc_df: pd.DataFrame) -> dict:
        """
        Calculate completion % per floor for heatmap visualization.
        
        Args:
            eqc_df: DataFrame with EQC data
        
        Returns:
            dict with completion rates per project/building/floor
        """
        df = self._filter_demo(eqc_df.copy())
        
        results = {
            'by_project': {}
        }
        
        # Define complete statuses
        complete_statuses = {'PASS', 'PASSED', 'APPROVED'}
        
        # Group by Project, Building (L1), Floor (L2)
        required_cols = ['Location L0', 'Location L1', 'Location L2', 'EQC Stage Status']
        if not all(col in df.columns for col in required_cols):
            return {'error': f'Missing required columns: {required_cols}'}
        
        df['__Complete'] = df['EQC Stage Status'].astype(str).str.upper().isin(complete_statuses)
        
        # Determine project
        if 'Project' in df.columns:
            proj_col = 'Project'
        elif 'Project Name' in df.columns:
            proj_col = 'Project Name'
        else:
            proj_col = 'Location L0'
        
        grouped = df.groupby([proj_col, 'Location L1', 'Location L2'])
        
        for (project, building, floor), group_df in grouped:
            project = canonicalize_project_name(str(project).strip())
            building = str(building).strip()
            floor = str(floor).strip()
            
            if not floor or floor in ('', 'nan', 'None'):
                continue
            
            total = len(group_df)
            complete = int(group_df['__Complete'].sum())
            pct = round(complete / total * 100, 1) if total > 0 else 0
            
            # Color coding
            if pct >= 80:
                color = 'green'
            elif pct >= 50:
                color = 'yellow'
            elif pct > 0:
                color = 'red'
            else:
                color = 'gray'
            
            if project not in results['by_project']:
                results['by_project'][project] = {}
            if building not in results['by_project'][project]:
                results['by_project'][project][building] = {}
            
            results['by_project'][project][building][floor] = {
                'total': total,
                'complete': complete,
                'percentage': pct,
                'color': color
            }
        
        return results
    
    # ========== REPEAT OFFENDERS ==========
    
    def analyze_repeat_offenders(self, issues_df: pd.DataFrame) -> dict:
        """
        Identify locations and contractors with most issues.
        
        Args:
            issues_df: DataFrame with Instructions data
        
        Returns:
            dict with top offenders by location, contractor, and type
        """
        df = self._filter_demo(issues_df.copy())
        
        # Filter out Safety issues (Quality only)
        if 'Type L0' in df.columns:
            safety_mask = df['Type L0'].astype(str).str.contains('safety', case=False, na=False)
            df = df[~safety_mask]
        
        results = {
            'by_location': [],
            'by_contractor': [],
            'by_issue_type': []
        }
        
        # By Location (Project + Building/Floor)
        if all(col in df.columns for col in ['Location L1', 'Location L2']):
            if 'Project' in df.columns:
                proj_col = 'Project'
            elif 'Project Name' in df.columns:
                proj_col = 'Project Name'
            else:
                proj_col = 'Location L0'

            df['__ProjectKey'] = df[proj_col].astype(str).apply(lambda x: canonicalize_project_name(x.strip()))
            df['__LocationKey'] = df['Location L1'].astype(str).str.strip() + ' / ' + df['Location L2'].astype(str).str.strip()

            grouped = df.groupby(['__ProjectKey', '__LocationKey'])
            rows = []
            for (project, loc), loc_df in grouped:
                if not str(loc).strip():
                    continue
                count = len(loc_df)
                closed = len(loc_df[loc_df['Current Status'].astype(str).str.upper().isin({'CLOSED', 'RESPONDED'})])
                closure_rate = round(closed / count * 100, 1) if count > 0 else 0
                rows.append({
                    'project': str(project).strip(),
                    'location': str(loc).strip(),
                    'count': int(count),
                    'closure_rate': closure_rate
                })

            # Sort by issue count descending and take top 10
            results['by_location'] = sorted(rows, key=lambda x: x['count'], reverse=True)[:10]
        
        # By Contractor/Team
        if 'Assigned Team' in df.columns:
            team_counts = df['Assigned Team'].value_counts().head(10)
            
            for team, count in team_counts.items():
                team_df = df[df['Assigned Team'] == team]
                closed = len(team_df[team_df['Current Status'].astype(str).str.upper().isin({'CLOSED', 'RESPONDED'})])
                closure_rate = round(closed / count * 100, 1) if count > 0 else 0
                
                results['by_contractor'].append({
                    'contractor': str(team),
                    'count': int(count),
                    'closure_rate': closure_rate
                })
        
        # By Issue Type
        if 'Type L0' in df.columns:
            type_counts = df['Type L0'].value_counts().head(10)
            
            for typ, count in type_counts.items():
                type_df = df[df['Type L0'] == typ]
                closed = len(type_df[type_df['Current Status'].astype(str).str.upper().isin({'CLOSED', 'RESPONDED'})])
                closure_rate = round(closed / count * 100, 1) if count > 0 else 0
                
                results['by_issue_type'].append({
                    'type': str(typ),
                    'count': int(count),
                    'closure_rate': closure_rate
                })
        
        return results
    
    # ========== POST-APPROVAL ISSUE DETECTION ==========
    
    def detect_post_approval_issues(self, eqc_df: pd.DataFrame, issues_df: pd.DataFrame) -> dict:
        """
        Detect cases where approved checklists still have issues raised.
        
        Args:
            eqc_df: DataFrame with EQC data
            issues_df: DataFrame with Instructions data
        
        Returns:
            dict with flagged post-approval issues grouped by project
        """
        eqc = self._filter_demo(eqc_df.copy())
        issues = self._filter_demo(issues_df.copy())
        
        # Filter out Safety issues (only check Quality issues)
        if 'Type L0' in issues.columns:
            safety_mask = issues['Type L0'].astype(str).str.contains('safety', case=False, na=False)
            issues = issues[~safety_mask]
        
        # Add project identification
        def get_project_name(row):
            for col in ['Project Name', 'Location L0', 'Project']:
                if col in row.index:
                    name = str(row[col]).strip()
                    if name and name not in ('', 'nan', 'None'):
                        return canonicalize_project_name(name)
            return 'Unknown'
        
        eqc['__Project'] = eqc.apply(get_project_name, axis=1)
        issues['__Project'] = issues.apply(get_project_name, axis=1)
        
        results = {
            'flagged_issues': [],
            'by_project': {},
            'summary': {}
        }
        
        # Filter approved EQC records
        approved_statuses = {'PASS', 'PASSED', 'APPROVED'}
        eqc_approved = eqc[eqc['Status'].astype(str).str.upper().isin(approved_statuses)].copy()
        
        # Parse dates
        def parse_date(s):
            s = str(s).strip()
            for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'):
                try:
                    return pd.to_datetime(s, format=fmt)
                except:
                    continue
            return None
        
        if 'Date' in eqc_approved.columns:
            eqc_approved['__ApprovalDate'] = eqc_approved['Date'].apply(parse_date)
        else:
            return {'error': 'Missing Date column in EQC data'}
        
        if 'Raised On Date' in issues.columns:
            issues['__RaisedDate'] = issues['Raised On Date'].apply(parse_date)
        else:
            return {'error': 'Missing Raised On Date column in Issues data'}
        
        # Filter to only OPEN/UNRESOLVED issues (ignore closed ones)
        open_statuses = {'RAISED', 'REJECTED'}
        if 'Current Status' in issues.columns:
            issues = issues[issues['Current Status'].astype(str).str.upper().isin(open_statuses)].copy()
        
        # Type matching rules
        def match_type(eqc_type: str, issue_type: str) -> bool:
            """Check if EQC type matches issue type."""
            eqc_lower = str(eqc_type).lower()
            issue_lower = str(issue_type).lower()
            
            type_mappings = [
                (['rcc', 'concrete', 'slab', 'column', 'beam', 'footing'], ['rcc', 'supervision poor', 'workmanship poor']),
                (['plaster', 'gypsum'], ['plaster', 'gypsum']),
                (['waterproof'], ['waterproof']),
                (['tiling', 'tile'], ['tiling']),
                (['painting'], ['painting']),
                (['electrical'], ['electrical']),
                (['plumbing'], ['plumbing']),
            ]
            
            for eqc_keywords, issue_keywords in type_mappings:
                if any(kw in eqc_lower for kw in eqc_keywords):
                    if any(kw in issue_lower for kw in issue_keywords):
                        return True
            
            return False
        
        # Cross-reference: Find issues raised AFTER each approved EQC at same location AND SAME PROJECT
        flagged_count = 0
        project_counts = defaultdict(int)
        
        for _, eqc_row in eqc_approved.iterrows():
            eqc_project = eqc_row['__Project']
            eqc_building = str(eqc_row.get('Location L1', '')).strip()
            eqc_floor = str(eqc_row.get('Location L2', '')).strip()
            eqc_type = str(eqc_row.get('Eqc Type', '')).strip()
            eqc_date = eqc_row['__ApprovalDate']
            
            if pd.isna(eqc_date) or not eqc_building or not eqc_floor or eqc_project == 'Unknown':
                continue
            
            # Find matching issues at SAME PROJECT, same location with same/matching type and raised AFTER approval
            project_issues = issues[issues['__Project'] == eqc_project]
            
            for _, issue_row in project_issues.iterrows():
                issue_building = str(issue_row.get('Location L1', '')).strip()
                issue_floor = str(issue_row.get('Location L2', '')).strip()
                issue_type = str(issue_row.get('Type L0', '')).strip()
                issue_date = issue_row['__RaisedDate']
                issue_status = str(issue_row.get('Current Status', '')).strip().upper()
                
                if pd.isna(issue_date):
                    continue
                
                # Verify issue status is OPEN/UNRESOLVED
                if issue_status not in {'RAISED', 'REJECTED'}:
                    continue
                
                # Check if same location, matching type, and issue raised AFTER approval
                if (eqc_building == issue_building and 
                    eqc_floor == issue_floor and
                    match_type(eqc_type, issue_type) and
                    issue_date > eqc_date):
                    
                    flagged_count += 1
                    project_counts[eqc_project] += 1
                    results['flagged_issues'].append({
                        'project': eqc_project,
                        'eqc_type': eqc_type,
                        'issue_type': issue_type,
                        'location': f"{eqc_building} / {eqc_floor}",
                        'eqc_approval_date': eqc_date.strftime('%d/%m/%Y'),
                        'issue_raised_date': issue_date.strftime('%d/%m/%Y'),
                        'days_after': (issue_date - eqc_date).days,
                        'issue_description': str(issue_row.get('Description', ''))[:100]
                    })
        
        # Group by project
        for issue in results['flagged_issues']:
            proj = issue['project']
            if proj not in results['by_project']:
                results['by_project'][proj] = []
            results['by_project'][proj].append(issue)
        
        results['summary'] = {
            'total_flagged': flagged_count,
            'approved_checklists': len(eqc_approved),
            'projects_affected': len(project_counts),
            'by_project_count': dict(project_counts)
        }
        
        # Sort by days after (most recent first)
        results['flagged_issues'] = sorted(results['flagged_issues'], key=lambda x: x['days_after'], reverse=True)[:50]
        
        return results


# Standalone functions for easy use
def analyze_overdue(issues_csv_path: str) -> dict:
    """Analyze overdue issues from CSV file."""
    try:
        df = pd.read_csv(issues_csv_path)
        analyzer = AdvancedAnalytics()
        return analyzer.analyze_overdue_issues(df)
    except Exception as e:
        return {'error': str(e)}


def analyze_floor_heatmap(eqc_csv_path: str) -> dict:
    """Generate floor heatmap data from CSV file."""
    try:
        df = pd.read_csv(eqc_csv_path)
        analyzer = AdvancedAnalytics()
        return analyzer.analyze_floor_completion(df)
    except Exception as e:
        return {'error': str(e)}


def analyze_offenders(issues_csv_path: str) -> dict:
    """Analyze repeat offenders from CSV file."""
    try:
        df = pd.read_csv(issues_csv_path)
        analyzer = AdvancedAnalytics()
        return analyzer.analyze_repeat_offenders(df)
    except Exception as e:
        return {'error': str(e)}


def detect_post_approval(eqc_csv_path: str, issues_csv_path: str) -> dict:
    """Detect post-approval issues from CSV files."""
    try:
        eqc_df = pd.read_csv(eqc_csv_path)
        issues_df = pd.read_csv(issues_csv_path)
        analyzer = AdvancedAnalytics()
        return analyzer.detect_post_approval_issues(eqc_df, issues_df)
    except Exception as e:
        return {'error': str(e)}
