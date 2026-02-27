import re
from typing import Optional
import pandas as pd


def canonicalize_project_name(name: Optional[str]) -> str:
    """Return a canonical project display name.

    Rules:
    - Normalize separators (underscore/hyphen) to spaces and collapse whitespace.
    - Map known variants to canonical names:
        * any contains 'city' and 'life' -> 'Itrend City Life'
        * any contains 'futura'         -> 'Itrend Futura'
        * any contains 'palacio'        -> 'Itrend Palacio'
    - For other projects, return a cleaned, consistently spaced Title Case name
      so variants like "abc_residency" and "ABC-Residency" unify.
    """
    s0 = str(name or '').strip()
    if not s0:
        return ''
    key = s0.lower().replace('_', ' ').replace('-', ' ')
    key = re.sub(r"\s+", " ", key).strip()

    # Canonical mappings for known projects
    if 'city' in key and 'life' in key:
        return 'Itrend City Life'
    if 'futura' in key:
        return 'Itrend Futura'
    if 'palacio' in key:
        return 'Itrend Palacio'
    if 'vesta' in key:
        return 'Itrend Vesta'
    if 'landmarc' in key and 'saheel' not in key:
        return 'Landmarc'
    if 'saheel' in key and 'landmarc' in key:
        return 'Saheel Landmarc'

    # Generic normalization for any other project
    tokens = [t for t in key.split(' ') if t]
    # Title-case but keep ALLCAPS tokens as-is (e.g., IT PARK)
    pretty = ' '.join(t if t.isupper() else t.capitalize() for t in tokens)
    return pretty


def canonical_project_from_row(row: pd.Series) -> str:
    """Derive project key from a dataframe row using common columns and canonicalize it.

    Preference order: Location L0 -> Project -> Project Name
    """
    for col in ("Location L0", "Project", "Project Name"):
        if col in row and str(row.get(col) or '').strip():
            return canonicalize_project_name(row.get(col))
    return ''
