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


_BUILDING_WORDS = {
    "wing",
    "tower",
    "building",
    "block",
    "bldg",
    "blk",
    "buliding",
    "biilding",
    "bellding",
    "bldng",
}

_SPECIAL_BUILDING_BUCKETS = {
    "development": "Development",
    "mlcp": "MLCP",
}


def building_identity(value: object) -> str:
    """Return the comparable building identifier from labels like A wing/Building A."""
    s = str(value or "").strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a", "unknown", "unknwn", "-", "--"}:
        return ""

    keywords = "|".join(sorted(_BUILDING_WORDS, key=len, reverse=True))
    patterns = [
        rf"\b({keywords})\b[\s\-\.]*([A-Za-z0-9]{{1,8}})\b",
        rf"\b([A-Za-z0-9]{{1,8}})\b[\s\-\.]*({keywords})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, s, re.I)
        if not m:
            continue
        g1, g2 = m.groups()
        ident = g2 if g1.lower() in _BUILDING_WORDS else g1
        ident = re.sub(r"[^A-Za-z0-9]+", "", ident).upper()
        if re.fullmatch(r"(?:[A-Z]\d?|\d+[A-Z]?)", ident):
            return ident

    return ""


def normalize_building_names_by_majority(
    df: pd.DataFrame,
    building_col: str = "Location L1",
    project_col: str = "__Project",
    unknown_label: str = "UNKNOWN",
) -> pd.DataFrame:
    """Normalize building labels to a canonical "Building <id>" form.

    Any label that contains a building identifier, such as "Tower A",
    "A wing", or "B tower", is rewritten to "Building A" or "Building B".
    Palacio rows are left untouched so their data stays exactly as-is.
    Labels that do not match a building identifier are mapped to UNKNOWN so
    building-wise reports only expose actual building buckets.
    """
    if building_col not in df.columns:
        return df

    work = df.copy()

    def _normalize(row: pd.Series) -> object:
        label = str(row.get(building_col) or "").strip()
        if not label or label.lower() in {"nan", "none", "null", "na", "n/a", "-", "--"}:
            return unknown_label

        project = canonical_project_from_row(row)
        if project == "Itrend Palacio":
            return label

        special = _SPECIAL_BUILDING_BUCKETS.get(label.lower())
        if special:
            return special

        ident = building_identity(label)
        if ident:
            return f"Building {ident}"

        return unknown_label

    work[building_col] = work.apply(_normalize, axis=1)
    return work
