"""
Combined Quality Dashboard - Project-wise summary
Shows both Quality Observations (Issues) and EQC data in one view.
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple
import re
from project_utils import canonicalize_project_name


def _filter_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DEMO/test projects."""
    demo_cols = ['Project', 'Project Name', 'Location L0']
    for col in demo_cols:
        if col in df.columns:
            mask = df[col].astype(str).str.contains('DEMO', case=False, na=False)
            df = df[~mask]
    return df


def _parse_date(s: str) -> date | None:
    """Parse date from various formats."""
    s = str(s).strip()
    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            continue
    return None


def _get_project_name(row: pd.Series) -> str:
    """Extract and canonicalize project name from row."""
    # Priority: Project Name > Location L0 > Project
    for col in ['Project Name', 'Location L0', 'Project']:
        if col in row.index:
            name = str(row[col]).strip()
            if name and name not in ('', 'nan', 'None'):
                # Use the shared canonicalization function for consistency
                return canonicalize_project_name(name)
    return 'Unknown'


def _is_external(row: pd.Series, external_names: str = "Omkar") -> bool:
    """Check if observation is External based on Raised By field.
    
    Matches analysis_issues.py logic for consistency:
    - External if Raised By exactly matches any external name (case-insensitive)
    - Support multiple names via comma or pipe separators
    - Default external names: "Omkar"
    """
    raised_by = str(row.get("Raised By", "")).strip().upper()
    
    # Parse external names (support comma or pipe separators)
    import re as _re
    names_raw = (external_names or "").strip()
    if not names_raw:
        return False
    
    tokens = [t.strip() for t in _re.split(r"[\|,]", names_raw) if t.strip()]
    up_tokens = [t.upper() for t in tokens]
    
    # Exact match (case-insensitive)
    for token in up_tokens:
        if raised_by == token:
            return True
    
    return False


def generate_quality_dashboard(eqc_df: pd.DataFrame, issues_df: pd.DataFrame, target_date: date = None, external_names: str = "Omkar") -> dict:
    """
    Generate combined quality dashboard data.
    
    Args:
        eqc_df: EQC CSV DataFrame
        issues_df: Instructions CSV DataFrame
        target_date: Target date for Today (defaults to today)
        external_names: Names to identify as external (comma or pipe separated, default: "Omkar")
    
    Returns:
        dict with project-wise observations and EQC data
    """
    if target_date is None:
        target_date = date.today()
    
    # Filter DEMO projects
    eqc = _filter_demo(eqc_df.copy())
    issues = _filter_demo(issues_df.copy())
    
    # Filter out Safety issues
    if 'Type L0' in issues.columns:
        safety_mask = issues['Type L0'].astype(str).str.contains('safety', case=False, na=False)
        issues = issues[~safety_mask]
    
    # Ensure required columns exist
    if 'Raised On Date' not in issues.columns:
        issues['Raised On Date'] = None
    if 'Raised By' not in issues.columns:
        issues['Raised By'] = ""
    if 'Current Status' not in issues.columns:
        issues['Current Status'] = ""
    
    # Parse dates
    if 'Raised On Date' in issues.columns:
        issues['__RaisedDate'] = issues['Raised On Date'].apply(_parse_date)
    
    if 'Date' in eqc.columns:
        eqc['__Date'] = eqc['Date'].apply(_parse_date)
    
    # Get project names
    issues['__Project'] = issues.apply(_get_project_name, axis=1)
    eqc['__Project'] = eqc.apply(_get_project_name, axis=1)
    
    # Determine Internal/External using the same logic as analysis_issues.py
    issues['__IsExternal'] = issues.apply(lambda row: _is_external(row, external_names), axis=1)
    
    # Get unique projects (from both)
    all_projects = sorted(set(issues['__Project'].unique()) | set(eqc['__Project'].unique()))
    all_projects = [p for p in all_projects if p and p != 'Unknown']
    
    # Build result
    result = {
        'date': target_date.strftime('%d/%m/%Y'),
        'projects': []
    }
    
    for proj in all_projects:
        proj_issues = issues[issues['__Project'] == proj]
        proj_eqc = eqc[eqc['__Project'] == proj]
        
        # Split Internal/External
        internal = proj_issues[~proj_issues['__IsExternal']]
        external = proj_issues[proj_issues['__IsExternal']]
        
        def count_observations(df: pd.DataFrame, date_col: str = '__RaisedDate') -> dict:
            """Count observations for a timeframe."""
            # Exact status matching as per analysis_issues.py
            open_statuses = {'RAISED', 'REJECTED'}
            closed_statuses = {'CLOSED', 'RESPONDED'}
            
            def get_counts(sub_df: pd.DataFrame) -> dict:
                if sub_df.empty:
                    return {'raised': 0, 'open': 0, 'closed': 0}
                statuses = sub_df['Current Status'].astype(str).str.upper()
                open_count = int(statuses.isin(open_statuses).sum())
                closed_count = int(statuses.isin(closed_statuses).sum())
                raised_count = open_count + closed_count  # Total raised = open + closed
                return {'raised': raised_count, 'open': open_count, 'closed': closed_count}
            
            # Today
            today_mask = df[date_col] == target_date if date_col in df.columns else pd.Series(False, index=df.index)
            today_counts = get_counts(df[today_mask])
            
            # This month
            month_mask = df[date_col].apply(lambda d: d and d.year == target_date.year and d.month == target_date.month if d else False)
            month_counts = get_counts(df[month_mask])
            
            # Cumulative (all-time)
            cum_counts = get_counts(df)
            
            return {
                'today': today_counts,
                'month': month_counts,
                'cumulative': cum_counts
            }
        
        def count_eqc(df: pd.DataFrame) -> dict:
            """Count EQC by stage."""
            if df.empty:
                return {'total': {'pre': 0, 'during': 0, 'post': 0}, 'today': {'pre': 0, 'during': 0, 'post': 0}}
            
            def stage_counts(sub: pd.DataFrame) -> dict:
                if sub.empty:
                    return {'pre': 0, 'during': 0, 'post': 0}
                s = sub.get('Stage', pd.Series()).astype(str).str.lower()
                pre = s.str.contains('pre', na=False).sum()
                during = s.str.contains('during', na=False).sum()
                post = s.str.contains('post', na=False).sum()
                reinf = s.str.contains('reinforce', na=False).sum()
                shut = s.str.contains('shutter', na=False).sum()
                other = len(s) - (pre + during + post + reinf + shut)
                
                # Cumulative logic: Pre = total, During = during+post+other, Post = post+other
                total = len(s)
                return {
                    'pre': total,
                    'during': int(during + post + other),
                    'post': int(post + other)
                }
            
            # Today
            today_mask = df['__Date'] == target_date if '__Date' in df.columns else pd.Series(False, index=df.index)
            today_df = df[today_mask]
            
            # For today, use raw counts
            def today_stage_counts(sub: pd.DataFrame) -> dict:
                if sub.empty:
                    return {'pre': 0, 'during': 0, 'post': 0}
                s = sub.get('Stage', pd.Series()).astype(str).str.lower()
                pre = s.str.contains('pre', na=False).sum()
                during = s.str.contains('during', na=False).sum()
                post = s.str.contains('post', na=False).sum()
                other = len(s) - (pre + during + post)
                return {'pre': int(pre), 'during': int(during), 'post': int(post + other)}
            
            return {
                'total': stage_counts(df),
                'today': today_stage_counts(today_df)
            }
        
        proj_data = {
            'name': proj,
            'observations': {
                'internal': count_observations(internal),
                'external': count_observations(external)
            },
            'eqc': count_eqc(proj_eqc)
        }
        
        result['projects'].append(proj_data)
    
    return result


def generate_dashboard_html_table(data: dict) -> str:
    """Generate HTML table from dashboard data."""
    html = f'''
    <div class="table-responsive">
        <table class="table table-bordered table-sm align-middle text-center">
            <thead>
                <tr class="table-primary">
                    <th colspan="12" class="text-center">Quality Observation Dashboard</th>
                    <th colspan="4" class="text-center">Quality EQC Dashboard</th>
                </tr>
                <tr class="table-light">
                    <th rowspan="2">Sr.No</th>
                    <th rowspan="2">Project Name</th>
                    <th rowspan="2">Type</th>
                    <th colspan="3" class="bg-info text-white">Today Observation</th>
                    <th colspan="3" class="bg-warning">This Month Report</th>
                    <th colspan="3" class="bg-danger text-white">Cumulative Report</th>
                    <th rowspan="2">Type</th>
                    <th colspan="3" class="bg-success text-white">Pre / During / Post</th>
                </tr>
                <tr class="table-light">
                    <th>Raised</th><th class="text-danger">Open</th><th class="text-success">Closed</th>
                    <th>Raised</th><th class="text-danger">Open</th><th class="text-success">Closed</th>
                    <th>Raised</th><th class="text-danger">Open</th><th class="text-success">Closed</th>
                    <th>Pre</th><th>During</th><th>Post</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for i, proj in enumerate(data['projects'], 1):
        obs = proj['observations']
        eqc = proj['eqc']
        
        # Internal row
        html += f'''
            <tr>
                <td rowspan="2">{i}</td>
                <td rowspan="2"><strong>{proj['name']}</strong></td>
                <td>Internal</td>
                <td>{obs['internal']['today']['raised']}</td>
                <td class="text-danger">{obs['internal']['today']['open']}</td>
                <td class="text-success">{obs['internal']['today']['closed']}</td>
                <td>{obs['internal']['month']['raised']}</td>
                <td class="text-danger">{obs['internal']['month']['open']}</td>
                <td class="text-success">{obs['internal']['month']['closed']}</td>
                <td>{obs['internal']['cumulative']['raised']}</td>
                <td class="text-danger">{obs['internal']['cumulative']['open']}</td>
                <td class="text-success">{obs['internal']['cumulative']['closed']}</td>
                <td>Total</td>
                <td>{eqc['total']['pre']}</td>
                <td>{eqc['total']['during']}</td>
                <td>{eqc['total']['post']}</td>
            </tr>
        '''
        
        # External row
        html += f'''
            <tr>
                <td>External</td>
                <td>{obs['external']['today']['raised']}</td>
                <td class="text-danger">{obs['external']['today']['open']}</td>
                <td class="text-success">{obs['external']['today']['closed']}</td>
                <td>{obs['external']['month']['raised']}</td>
                <td class="text-danger">{obs['external']['month']['open']}</td>
                <td class="text-success">{obs['external']['month']['closed']}</td>
                <td>{obs['external']['cumulative']['raised']}</td>
                <td class="text-danger">{obs['external']['cumulative']['open']}</td>
                <td class="text-success">{obs['external']['cumulative']['closed']}</td>
                <td>Today</td>
                <td>{eqc['today']['pre']}</td>
                <td>{eqc['today']['during']}</td>
                <td>{eqc['today']['post']}</td>
            </tr>
        '''
    
    html += '''
            </tbody>
        </table>
    </div>
    '''
    
    return html
