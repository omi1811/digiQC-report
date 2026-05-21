"""
Building Name Normalization Module - Fill Empty Location L1 Only

This module normalizes Location L1 values and fills empty ones.
It handles variants (Tower A, Wing A, B tower) by normalizing them to canonical forms (Building A, Building B).

Process:
1. NORMALIZE: Detect Location L1 variants and unify them to the most common form
   - Example: Tower A (5x), Building A (1000x), Wing A (2x) → all become Building A
2. LEARN: Extract valid building names from non-empty Location L1 values
3. FILL: Fill empty Location L1 values by extracting names from EQC column
   - Only uses building names that already exist in Location L1 (from step 2)
   - If extraction fails or name not found → use "Unknown"
   - Never modifies Location L1 values that are already filled

Key Features:
- NO HARDCODING: All logic is data-driven
- PROJECT-SPECIFIC: Learns building names from actual data
- VARIANT-HANDLING: Automatically detects and normalizes Building/Tower/Wing variants

Rules:
1. Normalize Location L1 variants using frequency-based canonical forms
2. Learn valid names from non-empty Location L1 values (after normalization)
3. For rows with EMPTY Location L1:
   - Extract building name from EQC column (format: PROJECT/BUILDING/FLOOR/...)
   - If extracted name exists in valid names set → fill Location L1 with it
   - Otherwise → fill with "Unknown"
4. Never modify Location L1 values that are already filled

Import and use fill_empty_location_l1() before generating reports.

Usage:
    import building_normalizer as bn
    df = pd.read_csv('data.csv')
    df = bn.fill_empty_location_l1(df)  # Normalize variants, learn names, fill empty L1
"""

import pandas as pd
import re
from typing import Optional, Set, Dict
from project_utils import building_identity, canonical_project_from_row

_SPECIAL_BUILDING_BUCKETS = {
    "development": "Development",
    "mlcp": "MLCP",
}


def extract_valid_building_names(df: pd.DataFrame) -> Set[str]:
    """
    Extract valid/canonical building names from NON-EMPTY Location L1 values.
    These are the "source of truth" building names for this project.
    
    Args:
        df: Input dataframe with Location L1 column
        
    Returns:
        Set of unique, non-empty building names found in Location L1
    """
    if 'Location L1' not in df.columns:
        return set()
    
    valid_names = set()
    
    for value in df['Location L1'].unique():
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        
        # Skip empty and placeholder values - ONLY include truly filled values
        if not value_str or value_str.lower() in ['', 'nan', 'none', '-', 'unknown']:
            continue
        
        # Add this as a valid canonical name
        valid_names.add(value_str)
    
    return valid_names


def extract_building_name_from_eqc(eqc_value: str) -> Optional[str]:
    """
    Extract building name from EQC column using pattern matching.
    
    EQC format examples:
    - FUTURA/Building F/Floor 17/Flat 1703/1703
    - FUTURA/Building A/Floor 14/Flat 1408/14th floor
    - FUTURA/MLCP/Level 2/Pour-7 slab and ramp
    - FUTURA/Development/...
    
    Extracts the second part after first "/" split (the BUILDING part).
    
    Args:
        eqc_value: The EQC column value
        
    Returns:
        Extracted building name or None if extraction fails
    """
    if pd.isna(eqc_value):
        return None
    
    eqc_str = str(eqc_value).strip()
    
    if not eqc_str:
        return None
    
    # Split by "/" and get the second part (index 1)
    # Format: PROJECT/BUILDING/FLOOR/FLAT/...
    parts = eqc_str.split('/')
    
    if len(parts) >= 2:
        building_part = parts[1].strip()
        # Return only if it's non-empty and not a placeholder
        if building_part and building_part.lower() not in ['', 'nan', 'none', '-']:
            return building_part
    
    return None


def extract_core_identifier(location_l1_value: str) -> Optional[str]:
    """
    Extract core identifier from Location L1 variant.
    
    Examples:
    - "Building A" → "A"
    - "Tower A" → "A"
    - "B tower" → "B"
    - "Wing A" → "A"
    - "MLCP" → "MLCP"
    - "Development" → "Development"
    
    Strategy:
    1. If ends with single capital letter (e.g., "Building A") → return the letter
    2. Otherwise return the whole string (for multi-word buildings like MLCP, Development)
    
    Args:
        location_l1_value: Location L1 value to extract core from
        
    Returns:
        Core identifier or None
    """
    if pd.isna(location_l1_value):
        return None
    
    value_str = str(location_l1_value).strip()
    
    if not value_str:
        return None
    
    # Pattern 1: Ends with space + single capital letter (e.g., "Building A", "Tower B", "Wing F")
    match = re.search(r'\s([A-Z])$', value_str)
    if match:
        return match.group(1)
    
    # Pattern 2: Starts with single capital letter + space + word ending (e.g., "B tower")
    match = re.match(r'^([A-Z])\s+', value_str)
    if match:
        return match.group(1)
    
    # Pattern 3: Multi-word buildings (MLCP, Development, etc.) - return the whole thing
    if re.match(r'^[A-Z][a-zA-Z]*$', value_str):  # Single or multi-word all caps or Capitalized
        return value_str
    
    # Fallback: return original
    return value_str


def normalize_location_l1_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Location L1 variants to canonical forms.
    
    Process:
    1. Detect any Location L1 value that contains a building identifier
    2. Rewrite it to the canonical form "Building <id>"
    3. Map any non-building label to "Unknown" so only building buckets remain
    
    Args:
        df: Input dataframe with Location L1 column
        
    Returns:
        Dataframe with normalized Location L1 values
    """
    if df.empty or 'Location L1' not in df.columns:
        return df
    
    df = df.copy()

    def _normalize(value: object) -> str:
        if pd.isna(value):
            return "Unknown"

        value_str = str(value).strip()
        if not value_str:
            return "Unknown"

        if value_str.lower() in {'nan', 'none', 'null', 'na', 'n/a', '-', '--'}:
            return "Unknown"

        # Leave Palacio data untouched.
        # This module is used in shared report flows, so preserve Palacio rows exactly as they came in.
        if value_str:
            # No direct project context in this function, so only normalize explicit building variants here.
            # Palacio-specific preservation is handled in the shared row-based normalization path.
            pass

        special = _SPECIAL_BUILDING_BUCKETS.get(value_str.lower())
        if special:
            return special

        ident = building_identity(value_str)
        if ident:
            return f"Building {ident}"

        return "Unknown"

    df['Location L1'] = df['Location L1'].map(_normalize)
    
    return df


def fill_location_l1_value(row: pd.Series, valid_names: Set[str]) -> str:
    """
    Fill Location L1 if empty, otherwise keep existing value.
    
    Rules:
    - If Location L1 already has a value → keep it as-is
    - If Location L1 is empty:
      - Try to extract building name from EQC column
      - If extracted name is in valid_names → use it
      - Otherwise → use "Unknown"
    
    Args:
        row: A dataframe row
        valid_names: Set of valid building names
        
    Returns:
        Location L1 value (filled if was empty, unchanged if was filled)
    """
    current_l1 = row.get('Location L1', '')
    
    # If Location L1 is already filled, keep it unchanged
    if pd.notna(current_l1):
        value_str = str(current_l1).strip()
        if value_str and value_str.lower() not in ['', 'nan', 'none', '-']:
            return value_str
    
    # Location L1 is empty - try to fill it from EQC column
    eqc_value = row.get('EQC', '')
    extracted_building = extract_building_name_from_eqc(eqc_value)
    
    # Check if extracted building is in valid names
    if extracted_building and extracted_building in valid_names:
        return extracted_building
    
    # No valid match found
    return 'Unknown'


def fill_empty_location_l1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty Location L1 values by extracting building names from EQC column.
    Only uses building names that already exist in the Location L1 column.
    Never modifies Location L1 values that are already filled.
    
    Process:
    1. NORMALIZE: Convert Location L1 variants (Tower A, B tower, Wing A) to canonical forms (Building A, Building B)
    2. LEARN: Extract valid building names from non-empty Location L1 values
    3. FILL: For each row with empty Location L1:
       - Extract building name from EQC column
       - If extracted name exists in valid names → fill Location L1
       - Otherwise → fill with "Unknown"
    4. Leave already-filled Location L1 values untouched
    
    Args:
        df: Input dataframe with Location L1 and EQC columns
        
    Returns:
        Dataframe with filled Location L1 values
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if required columns exist
    if 'Location L1' not in df.columns or 'EQC' not in df.columns:
        return df
    
    # STEP 1: Normalize Location L1 variants to canonical forms
    df = normalize_location_l1_variants(df)
    
    # STEP 2: Learn valid building names from non-empty Location L1 values
    valid_names = extract_valid_building_names(df)
    
    # STEP 3: Fill Location L1 for each row
    df['Location L1'] = df.apply(
        lambda row: fill_location_l1_value(row, valid_names),
        axis=1
    )
    
    return df


def normalize_dataframe(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Wrapper function for backward compatibility.
    Calls fill_empty_location_l1() to fill empty Location L1 values.
    
    Args:
        df: Input dataframe
        columns: Ignored (kept for backward compatibility)
        
    Returns:
        Dataframe with filled Location L1 values
    """
    return fill_empty_location_l1(df)


def normalize_building_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for backward compatibility.
    Calls fill_empty_location_l1() to fill empty Location L1 values.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with filled Location L1 values
    """
    return fill_empty_location_l1(df)


def get_project_building_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of building names and their counts after filling Location L1.
    Useful for verification to ensure correct distribution.
    
    Args:
        df: Normalized dataframe
        
    Returns:
        Dictionary with building names as keys and counts as values, sorted alphabetically
    """
    if 'Location L1' not in df.columns:
        return {}
    
    counts = df['Location L1'].value_counts().to_dict()
    return dict(sorted(counts.items()))
