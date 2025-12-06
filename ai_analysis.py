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
        Analyze Issues CSV to find most repeating issues.
        
        Args:
            df: DataFrame with Issues data (columns: Type L0, Description, Current Status, Location)
        
        Returns:
            dict with issue patterns and top repeating issues
        """
        # Filter out DEMO projects
        df = self._filter_demo_projects(df)
        
        # Also filter out Safety issues (only Quality)
        if 'Type L0' in df.columns:
            safety_mask = df['Type L0'].astype(str).str.contains('safety', case=False, na=False)
            df = df[~safety_mask]
        
        results = {
            'summary': {},
            'by_category': {},
            'top_issues': [],
            'issue_clusters': []
        }
        
        # Ensure required columns exist
        if 'Type L0' not in df.columns:
            return {'error': 'Missing required column: Type L0'}
        
        # Overall summary
        total = len(df)
        status_col = 'Current Status' if 'Current Status' in df.columns else None
        
        if status_col:
            raised = len(df[df[status_col] == 'RAISED'])
            closed = len(df[df[status_col] == 'CLOSED'])
            results['summary'] = {
                'total_issues': total,
                'raised': raised,
                'closed': closed,
                'closure_rate': round(closed / total * 100, 1) if total > 0 else 0
            }
        else:
            results['summary'] = {'total_issues': total}
        
        # Analysis by Category (Type L0)
        category_counts = df['Type L0'].value_counts().head(10)
        for category, count in category_counts.items():
            results['by_category'][category] = {
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            }
        
        # Find most repeating issues using Description
        if 'Description' in df.columns:
            description_counts = df['Description'].value_counts().head(20)
            
            for desc, count in description_counts.items():
                if pd.isna(desc) or desc.strip() == '':
                    continue
                    
                # Get locations for this issue
                issue_df = df[df['Description'] == desc]
                locations = issue_df['Location L0'].unique()[:3] if 'Location L0' in df.columns else []
                
                results['top_issues'].append({
                    'description': desc[:100],  # Truncate long descriptions
                    'count': int(count),
                    'locations': list(locations)
                })
            
            # Semantic clustering of similar issues (if AI available)
            if self.use_ai and len(df) > 0:
                results['issue_clusters'] = self._cluster_similar_issues(df['Description'].dropna().unique())
        
        return results
    
    def _cluster_similar_issues(self, descriptions: list, threshold: float = 0.85) -> list:
        """Cluster semantically similar issue descriptions using embeddings."""
        if not self.use_ai or len(descriptions) < 2:
            return []
        
        try:
            # Limit to first 100 unique descriptions for performance
            descriptions = list(descriptions)[:100]
            
            # Get embeddings
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
                        'main': desc[:80],
                        'similar': [descriptions[j][:80] for j in similar_indices if j != i][:5],
                        'count': len(similar_indices)
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
        return analyzer.analyze_eqc_completion(df)
    except Exception as e:
        return {'error': str(e)}


def analyze_issues_file(filepath: str, use_ai: bool = True) -> dict:
    """Analyze an Issues CSV file."""
    try:
        df = pd.read_csv(filepath)
        analyzer = AIAnalyzer(use_ai=use_ai)
        return analyzer.analyze_issues_patterns(df)
    except Exception as e:
        return {'error': str(e)}


def standardize_checklists_file(filepath: str, use_ai: bool = True) -> dict:
    """Standardize checklist names in an EQC CSV file."""
    try:
        df = pd.read_csv(filepath)
        analyzer = AIAnalyzer(use_ai=use_ai)
        return analyzer.standardize_checklist_names(df)
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
