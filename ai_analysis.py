"""
AI Analysis Module for digiQC Report
Uses free, local NLP libraries for semantic analysis:
- sentence-transformers for semantic similarity
- rapidfuzz for fuzzy matching fallback
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import re

# Optional: Import sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using fallback methods.")

# Optional: Import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class AIAnalyzer:
    """AI-powered CSV analysis using free, local NLP libraries."""
    
    def __init__(self, use_ai=True):
        self.use_ai = use_ai and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        if self.use_ai:
            try:
                # Load lightweight model (22MB, runs locally)
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load AI model: {e}")
                self.use_ai = False
    
    def _filter_demo_projects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove DEMO/test projects from analysis."""
        demo_cols = ['Project', 'Project Name', 'Location L0']
        for col in demo_cols:
            if col in df.columns:
                mask = df[col].astype(str).str.contains('DEMO', case=False, na=False)
                df = df[~mask]
        return df
    
    # ========== EQC COMPLETION ANALYSIS ==========
    
    def analyze_eqc_completion(self, df: pd.DataFrame) -> dict:
        """
        Analyze EQC completion percentages with exceptions.
        
        Args:
            df: DataFrame with EQC data (columns: Location L0-L4, EQC Stage Status, Eqc Type, Stage)
        
        Returns:
            dict with completion stats per location hierarchy
        """
        # Filter out DEMO projects
        df = self._filter_demo_projects(df)
        
        results = {
            'summary': {},
            'by_building': {},
            'by_floor': {},
            'exceptions': []
        }
        
        # Ensure required columns exist
        required_cols = ['Location L0', 'Location L1', 'EQC Stage Status']
        if not all(col in df.columns for col in required_cols):
            return {'error': f'Missing required columns. Need: {required_cols}'}
        
        # Define completion statuses
        complete_statuses = {'PASS', 'APPROVED'}
        pending_statuses = {'APPROVAL_PENDING', 'IN_PROGRESS'}
        failed_statuses = {'REDO', 'FAIL'}
        
        # Overall summary
        total = len(df)
        completed = len(df[df['EQC Stage Status'].isin(complete_statuses)])
        pending = len(df[df['EQC Stage Status'].isin(pending_statuses)])
        failed = len(df[df['EQC Stage Status'].isin(failed_statuses)])
        
        results['summary'] = {
            'total_checks': total,
            'completed': completed,
            'pending': pending,
            'failed': failed,
            'completion_rate': round(completed / total * 100, 1) if total > 0 else 0
        }
        
        # Analysis by Building (Location L1)
        for building in df['Location L1'].dropna().unique():
            building_df = df[df['Location L1'] == building]
            b_total = len(building_df)
            b_completed = len(building_df[building_df['EQC Stage Status'].isin(complete_statuses)])
            
            results['by_building'][building] = {
                'total': b_total,
                'completed': b_completed,
                'completion_rate': round(b_completed / b_total * 100, 1) if b_total > 0 else 0
            }
        
        # Analysis by Floor (grouped by Building + Floor)
        if 'Location L2' in df.columns:
            floor_groups = df.groupby(['Location L1', 'Location L2'])
            for (building, floor), floor_df in floor_groups:
                if pd.isna(floor) or floor == '':
                    continue
                    
                f_total = len(floor_df)
                f_completed = len(floor_df[floor_df['EQC Stage Status'].isin(complete_statuses)])
                f_pending = len(floor_df[floor_df['EQC Stage Status'].isin(pending_statuses)])
                
                key = f"{building}/{floor}"
                results['by_floor'][key] = {
                    'building': building,
                    'floor': floor,
                    'total': f_total,
                    'completed': f_completed,
                    'pending': f_pending,
                    'completion_rate': round(f_completed / f_total * 100, 1) if f_total > 0 else 0
                }
                
                # Find exceptions (floors with low completion)
                if f_total > 0 and f_completed / f_total < 0.5:
                    # Get specific missing items
                    pending_items = floor_df[floor_df['EQC Stage Status'].isin(pending_statuses)]
                    if len(pending_items) > 0:
                        eqc_types = pending_items['Eqc Type'].unique()[:3]  # Top 3 pending types
                        results['exceptions'].append({
                            'location': key,
                            'completion_rate': round(f_completed / f_total * 100, 1),
                            'pending_count': f_pending,
                            'pending_types': list(eqc_types)
                        })
        
        # Sort exceptions by completion rate (lowest first)
        results['exceptions'] = sorted(results['exceptions'], key=lambda x: x['completion_rate'])[:10]
        
        return results
    
    # ========== ISSUES PATTERN ANALYSIS ==========
    
    def analyze_issues_patterns(self, df: pd.DataFrame) -> dict:
        """
        Analyze Issues CSV to find top 7-10 teams with most issues,
        then show their most repetitive instructions.
        
        Args:
            df: DataFrame with Issues data (columns: Description, Assigned Team)
        
        Returns:
            dict with top teams and their repetitive instructions
        """
        # Filter out DEMO projects
        df = self._filter_demo_projects(df)
        
        # Also filter out Safety issues (only Quality)
        if 'Type L0' in df.columns:
            safety_mask = df['Type L0'].astype(str).str.contains('safety', case=False, na=False)
            df = df[~safety_mask]
        
        # Ensure required columns exist
        if 'Assigned Team' not in df.columns:
            return {'error': 'Missing required column: Assigned Team'}
        if 'Description' not in df.columns:
            return {'error': 'Missing required column: Description'}
        
        results = {
            'teams': [],
            'note': f'Analyzed {len(df)} total instructions'
        }
        
        # Get top 7-10 teams by issue count
        team_counts = df['Assigned Team'].value_counts()
        # Filter out NaN teams and take top 10 (user will see 7-10)
        team_counts = team_counts[team_counts.index.notna()].head(10)
        
        for team_name in team_counts.index:
            team_df = df[df['Assigned Team'] == team_name]
            
            # Get top 5 repetitive instructions for this team
            description_counts = team_df['Description'].value_counts().head(5)
            
            team_issues = []
            for desc, count in description_counts.items():
                if pd.isna(desc) or str(desc).strip() == '':
                    continue
                
                desc_str = str(desc).strip()
                team_issues.append({
                    'description': desc_str[:100],  # Truncate long descriptions
                    'count': int(count)
                })
            
            # Only add team if it has issues
            if team_issues:
                results['teams'].append({
                    'team': str(team_name),
                    'total_issues': int(len(team_df)),
                    'top_instructions': team_issues
                })
        
        # Sanitize results recursively to ensure JSON-serializable values
        def _sanitize(v):
            # Handle pandas/numpy scalar types
            if isinstance(v, (np.int64, np.int32, np.integer)):
                return int(v)
            if isinstance(v, (np.float64, np.float32, np.floating)):
                if np.isnan(v):
                    return None
                return float(v)
            # If it's a sequence/array, sanitize each item
            if isinstance(v, (list, tuple, np.ndarray)):
                return [_sanitize(x) for x in list(v)]
            # If dict, sanitize recursively
            if isinstance(v, dict):
                return {str(k): _sanitize(val) for k, val in v.items()}
            # Scalar NaN/NA
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            
            return v

        return _sanitize(results)
    
    def _cluster_similar_issues(self, descriptions: list, threshold: float = 0.85, max_items: int = 50) -> list:
        """Cluster semantically similar issue descriptions using embeddings.
        
        Args:
            descriptions: List of description strings to cluster
            threshold: Similarity threshold for grouping (0-1)
            max_items: Maximum items to process (for performance)
        
        Returns:
            List of clusters, sorted by count descending
        """
        if not self.use_ai or len(descriptions) < 2:
            return []
        
        try:
            # Limit descriptions for performance
            descriptions = list(descriptions)[:max_items]
            if len(descriptions) < 2:
                return []
            
            # Get embeddings with batch processing for better performance
            try:
                embeddings = self.model.encode(descriptions, show_progress_bar=False, batch_size=32)
            except TypeError:
                # Fallback for older versions
                embeddings = self.model.encode(descriptions)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Group similar descriptions
            clusters = []
            used = set()
            
            for i, desc in enumerate(descriptions):
                if i in used:
                    continue
                    
                # Find similar descriptions
                similar_indices = np.where(similarity_matrix[i] > threshold)[0]
                if len(similar_indices) > 1:
                    cluster = {
                        'main': str(desc)[:80],
                        'similar': [str(descriptions[j])[:80] for j in similar_indices if j != i][:5],
                        'count': int(len(similar_indices))
                    }
                    clusters.append(cluster)
                    used.update(similar_indices)
            
            return sorted(clusters, key=lambda x: x['count'], reverse=True)[:5]
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return []
    
    # ========== CHECKLIST STANDARDIZATION ==========
    
    def standardize_checklist_names(self, df: pd.DataFrame, 
                                     mapping_file: str = 'checklist_mapping.json') -> dict:
        """
        Standardize EQC checklist names by removing contractor names.
        
        Phase 1: Use Team column to strip contractor names
        Phase 2: Use semantic similarity to group similar names
        
        Args:
            df: DataFrame with Eqc Type and Team columns
            mapping_file: Path to JSON file for caching mappings
        
        Returns:
            dict with original->standardized name mappings
        """
        results = {
            'mappings': {},
            'suggested_groups': [],
            'stats': {}
        }
        
        if 'Eqc Type' not in df.columns:
            return {'error': 'Missing required column: Eqc Type'}
        
        # Get unique checklist names
        unique_names = df['Eqc Type'].dropna().unique()
        results['stats']['total_unique'] = len(unique_names)
        
        # Load existing mappings if available
        existing_mappings = {}
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    existing_mappings = json.load(f)
            except:
                pass
        
        # Phase 1: Strip contractor names using Team column
        if 'Team' in df.columns:
            team_names = df['Team'].dropna().unique()
            
            for name in unique_names:
                if name in existing_mappings:
                    results['mappings'][name] = existing_mappings[name]
                    continue
                
                cleaned = name
                # Try to strip team names from the end
                for team in team_names:
                    if team and cleaned.endswith(team):
                        cleaned = cleaned[:-len(team)].strip()
                        # Clean trailing punctuation
                        cleaned = re.sub(r'[\s\.\-\:]+$', '', cleaned)
                        break
                
                # Additional cleanup patterns
                cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # Double spaces
                cleaned = cleaned.strip()
                
                if cleaned != name:
                    results['mappings'][name] = cleaned
        
        # Phase 2: Suggest semantic groupings (if AI available)
        if self.use_ai:
            cleaned_names = [results['mappings'].get(n, n) for n in unique_names]
            results['suggested_groups'] = self._suggest_checklist_groups(cleaned_names)
        
        results['stats']['mappings_created'] = len(results['mappings'])
        
        return results
    
    def _suggest_checklist_groups(self, names: list, threshold: float = 0.80) -> list:
        """Suggest groups of similar checklist names."""
        if not self.use_ai or len(names) < 2:
            return []
        
        try:
            # Get embeddings
            embeddings = self.model.encode(names)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find groups of similar names
            groups = []
            used = set()
            
            for i, name in enumerate(names):
                if i in used:
                    continue
                    
                similar_indices = np.where(similarity_matrix[i] > threshold)[0]
                if len(similar_indices) > 1:
                    group_names = [names[j] for j in similar_indices]
                    # Suggest shortest as canonical name
                    canonical = min(group_names, key=len)
                    groups.append({
                        'canonical': canonical,
                        'variations': [n for n in group_names if n != canonical][:5],
                        'count': len(similar_indices)
                    })
                    used.update(similar_indices)
            
            return sorted(groups, key=lambda x: x['count'], reverse=True)[:10]
            
        except Exception as e:
            print(f"Grouping error: {e}")
            return []


# ========== STANDALONE FUNCTIONS ==========

def analyze_eqc_file(filepath: str, use_ai: bool = True) -> dict:
    """Analyze an EQC CSV file."""
    try:
        df = pd.read_csv(filepath)
        analyzer = AIAnalyzer(use_ai=use_ai)
        result = analyzer.analyze_eqc_completion(df)
        return _sanitize_for_json(result)
    except Exception as e:
        return {'error': str(e)}


def analyze_issues_file(filepath: str, use_ai: bool = True) -> dict:
    """Analyze an Issues CSV file."""
    try:
        df = pd.read_csv(filepath)
        analyzer = AIAnalyzer(use_ai=use_ai)
        result = analyzer.analyze_issues_patterns(df)
        return _sanitize_for_json(result)
    except Exception as e:
        return {'error': str(e)}


def standardize_checklists_file(filepath: str, use_ai: bool = True) -> dict:
    """Standardize checklist names in an EQC CSV file."""
    try:
        df = pd.read_csv(filepath)
        analyzer = AIAnalyzer(use_ai=use_ai)
        result = analyzer.standardize_checklist_names(df)
        return _sanitize_for_json(result)
    except Exception as e:
        return {'error': str(e)}


# ========== CLI INTERFACE ==========

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Analysis for digiQC CSVs')
    parser.add_argument('file', help='Path to CSV file')
    parser.add_argument('--type', choices=['eqc', 'issues', 'standardize'], 
                        default='eqc', help='Type of analysis')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI features')
    
    args = parser.parse_args()
    
    use_ai = not args.no_ai
    
    if args.type == 'eqc':
        result = analyze_eqc_file(args.file, use_ai=use_ai)
    elif args.type == 'issues':
        result = analyze_issues_file(args.file, use_ai=use_ai)
    else:
        result = standardize_checklists_file(args.file, use_ai=use_ai)
    
    print(json.dumps(result, indent=2))


def _sanitize_for_json(v):
    """Recursively convert numpy/pandas scalars and NaNs to JSON-serializable Python types.

    Returns a new value with safe types (int, float, str, None, list, dict).
    """
    # Handle dicts
    if isinstance(v, dict):
        return {str(k): _sanitize_for_json(val) for k, val in v.items()}
    # Sequences/arrays
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_sanitize_for_json(x) for x in list(v)]
    # numpy int
    if isinstance(v, (np.integer,)):
        return int(v)
    # numpy floats
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    # pandas NA/NaN
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v
