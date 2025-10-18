#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd

from project_utils import canonical_project_from_row
import analysis_eqc as EQC
from openpyxl import load_workbook
from openpyxl.styles import Border, Side, Alignment

app = Flask(__name__)

# --- Helpers ---
CATEGORY_ORDER: List[Tuple[str, List[str]]] = [
    (
    "1. RCC / Structural Stage ",
        [
            # structural/aluform
            r"\brcc\b|reinforc|shuttering|\bstruct|slab|column|beam|footing",
            r"aluform|internal\s*handover",
        ],
    ),
    (
        "2. Masonry & Surface Preparation",
        [
            r"aac|autoclaved|block\s*masonry|fly\s*ash|brick\s*work",
            r"internal\s*plaster|plaster\s*works|gypsum",
        ],
    ),
    (
        "3. Waterproofing",
        [
            r"waterproof.*toilet|skirting|boxtype|toilet\s*&\s*skirting",
            r"brick\s*bat",
            r"pu\s*waterproof",
        ],
    ),
    (
        "4. Flooring & Finishes",
        [
            r"tiling.*dado|dado\s*\(toilet|kitchen\)|kitchen\s*dado",
            r"tiling.*floor|flooring",
            r"kitchen\s*platform|sink",
            r"granite.*(frame|door|window)",
            r"staircase.*skirting",
        ],
    ),
    (
        "5. Carpentry & Metal Works",
        [
            r"carpentry.*(door\s*frame|shutters?)",
            r"(door\s*frame|shutters?)",
            r"aluminium.*sliding|aluminum.*sliding|sliding\s*door|sliding\s*window",
            r"railings?\s*&?\s*grills?|ms\s*grill",
            r"ss\s*railing",
        ],
    ),
    (
        "6. MEP (Mechanical, Electrical & Plumbing)",
        [
            r"internal\s*plumbing|plumbing\s*works",
            r"drainage.*pvc|bathroom.*drain|kitchen.*drain",
            r"electrical.*conduit|wall.*conduiting|wiring",
            r"fire\s*fight|sprinkler",
            r"mep.*before.*casting|sleeve|opening",
        ],
    ),
    (
        "7. Painting & Finishing",
        [
            r"painting.*internal",
            r"painting.*(railing|grills).*oil",
        ],
    ),
    (
        "8. Final Cleaning & Handover",
        [
            r"cleaning.*acid|deep\s*clean",
        ],
    ),
]

UNCATEGORIZED_LABEL = "Uncategorized / Other"


def map_checklist_to_category(name: str) -> Tuple[int, str]:
    import re as _re
    s = (name or "").lower()
    for idx, (label, patterns) in enumerate(CATEGORY_ORDER):
        for pat in patterns:
            try:
                if _re.search(pat, s, _re.I):
                    return idx, label
            except Exception:
                continue
    return len(CATEGORY_ORDER), UNCATEGORIZED_LABEL


def robust_read_eqc(path: str) -> pd.DataFrame:
    last_err = None
    for sep in ("\t", ",", None):
        try:
            if sep is None:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
            else:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine="python")
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Failed to read EQC file")


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    # Drop DEMO rows
    cols = [c for c in ("Project", "Project Name", "Location L0") if c in df.columns]
    if cols:
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= df[c].astype(str).str.contains("DEMO", case=False, na=False)
        df = df[~mask]
    # Stage normalization & dates
    if "Stage" in df.columns:
        df["__Stage"] = df["Stage"].map(EQC.normalize_stage)
    if "Date" in df.columns:
        df["__Date"] = df["Date"].astype(str).map(EQC._parse_date_safe)
    else:
        df["__Date"] = None
    # Canonical project
    df["__Project"] = df.apply(canonical_project_from_row, axis=1)
    # Build a mapping of (Project, Building Letter) -> preferred label from CSV (preserve wording like Wing/Tower/Building)
    import re as _re_map
    bldg_pref_map: Dict[Tuple[str, str], str] = {}
    pref_counts: Dict[Tuple[str, str, str], int] = {}
    if "Location L1" in df.columns:
        for _, r in df.iterrows():
            proj = str(r.get("__Project", "") or "").strip()
            l1 = str(r.get("Location L1", "") or "").strip()
            if not proj or not l1:
                continue
            m = _re_map.search(r"\b(wing|tower|building|block|bldg|blk)\b[\s\-]*([A-Za-z0-9])\b", l1, _re_map.I) or \
                _re_map.search(r"\b([A-Za-z0-9])\b[\s\-]*\b(wing|tower|building|block|bldg|blk)\b", l1, _re_map.I)
            if m:
                g = m.groups()
                # extract the single-letter/number token regardless of order
                letter = g[1] if g[0].lower() in ("wing","tower","building","block","bldg","blk") else g[0]
                key = (proj, str(letter).upper())
                pref_counts[(proj, str(letter).upper(), l1)] = pref_counts.get((proj, str(letter).upper(), l1), 0) + 1
        # choose most frequent label for each (project, letter)
        for (proj, letter, label), cnt in pref_counts.items():
            cur = bldg_pref_map.get((proj, letter))
            if cur is None or cnt > pref_counts.get((proj, letter, cur), 0):
                bldg_pref_map[(proj, letter)] = label

    # Infer Building/Floor/Flat: prefer Location L1/L2/L3, but fall back to EQC fields (e.g., Eqc Type) if missing
    def infer_bld_floor_flat(row: pd.Series) -> Tuple[str, str, str]:
        import re as _re
        b = str(row.get("Location L1", "") or "").strip()
        f = str(row.get("Location L2", "") or "").strip()
        fl = str(row.get("Location L3", "") or "").strip()
        # Candidate source strings to search for fallbacks
        sources = []
        for c in ("EQC", "Eqc", "Eqc Type", "Stage", "Status"):
            if c in row and str(row[c]).strip():
                sources.append(str(row[c]))
        # Also include any column that looks like EQC path
        for c in row.index:
            sc = str(c).strip().lower()
            if sc in {"eqc path", "eqc location", "eqc name"} and str(row[c]).strip():
                sources.append(str(row[c]))
        # Tokenize sources for targeted heuristics
        tokens: List[str] = []
        for s in sources:
            tokens.extend([t.strip() for t in _re.split(r"[>/|,;:\-–—]+", str(s)) if t.strip()])
        # Building fallback patterns (A wing, Building A, Tower B, Block C)
        if not b:
            patt_building = [
                _re.compile(r"\b(wing|tower|building|block)[\s\-]*([A-Za-z0-9])\b", _re.I),
                _re.compile(r"\b([A-Za-z0-9])[\s\-]*(wing|tower|building|block)\b", _re.I),
                _re.compile(r"\b(bldg|blk)\.?[\s\-]*([A-Za-z0-9])\b", _re.I),
                # do not use single-letter fallback unless no other hint
            ]
            proj = str(row.get("__Project", "") or "").strip()
            letter_found = None
            type_found = None
            for s in sources + tokens:
                for pat in patt_building:
                    m = pat.search(s)
                    if m:
                        # Normalize to 'A wing'
                        if len(m.groups()) == 2:
                            g1, g2 = m.groups()
                            type_token = g1 if g1.lower() in ("wing","tower","building","block","bldg","blk") else g2
                            letter_token = g2 if type_token == g1 else g1
                            letter_found = str(letter_token).strip().upper()
                            type_found = str(type_token).strip().lower()
                            break
                if b:
                    break
            if not b and letter_found:
                # Prefer exact label used elsewhere in this project for this letter
                pref = bldg_pref_map.get((proj, letter_found))
                if pref:
                    b = pref
                elif type_found:
                    # Expand type token to a nice word
                    type_map = {"bldg": "Building", "blk": "Block"}
                    t = type_map.get(type_found, type_found).title()
                    b = f"{t} {letter_found}"
        # Floor from Location fields first (L0-L4), else fallback
        if not f:
            # Try Location L2 or L3 text for floor-like tokens
            for loc_field in ("Location L2", "Location L3", "Location L4", "Location L1", "Location L0"):
                sv = str(row.get(loc_field, "") or "").strip()
                if not sv:
                    continue
                # Recognize special non-flat structures
                if _re.search(r"\bshear\s*wall\b|stair\s*case|lobby|podium|development|terrace|parking", sv, _re.I):
                    f = sv
                    break
                m = _re.search(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", sv, _re.I) or _re.search(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", sv, _re.I)
                if m:
                    num = m.group(m.lastindex or 1)
                    f = f"Floor {num}"
                    break
        # Floor fallback patterns (Floor 10, 10th Floor, Level 5) from EQC if still missing
        if not f:
            patt_floor = [
                _re.compile(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", _re.I),
                _re.compile(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", _re.I),
                _re.compile(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", _re.I),
                _re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", _re.I),
            ]
            for s in sources + tokens:
                for pat in patt_floor:
                    m = pat.search(s)
                    if m:
                        # pick the last numeric group found
                        num = m.group(m.lastindex or 1)
                        f = f"Floor {num}"
                        break
                if f:
                    break
        # Flat fallback patterns (Flat 101, Shop 065, Unit 902A, Apartment 12)
        if not fl:
            # Require explicit keywords to avoid misreading product codes like "Samshield XL 1500" as a flat
            patt_flat = [
                _re.compile(r"\b(flat|unit|apt|apartment|shop)[\s\-]*([A-Z]?\d{1,4}[A-Z]?)\b", _re.I),
                _re.compile(r"\b(room)[\s\-]*(\d{1,4}[A-Z]?)\b", _re.I),
            ]
            for s in sources + tokens:
                for pat in patt_flat:
                    m = pat.search(s)
                    if m:
                        val = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                        # Normalize leading zeros: 065 -> 65
                        core = _re.sub(r"^([A-Z]?)(0+)(\d)", r"\1\3", val)
                        # Respect type if specified (Flat vs Shop)
                        typ = m.group(1).strip().title() if (m.lastindex and m.lastindex >= 2) else "Flat"
                        if typ.lower() not in ("flat","unit","apt","apartment","shop"):
                            typ = "Flat"
                        # Unit/Apt/Apartment -> Flat
                        if typ.lower() in ("unit","apt","apartment"):
                            typ = "Flat"
                        fl = f"{typ} {core}"
                        break
                if fl:
                    break
        # Standardize labels
        def _std_building(x: str) -> str:
            # Preserve CSV wording exactly if it came from Location L1
            return (x or "").strip()

        def _std_floor(x: str) -> str:
            s = (x or "").strip()
            if not s:
                return ""
            m = _re.search(r"(\d{1,2})", s)
            if m and _re.search(r"floor|fl|level|lvl|^\d{1,2}$", s, _re.I):
                return f"Floor {m.group(1)}"
            return s

        def _std_flat(x: str) -> str:
            s = (x or "").strip()
            if not s:
                return ""
            if _re.search(r"^flat\s+", s, _re.I):
                return _re.sub(r"^flat\s+", "Flat ", s, flags=_re.I)
            m = _re.search(r"([A-Z]?\d{1,4}[A-Z]?)", s)
            if m:
                # Keep Shop vs Flat if present
                t = "Flat"
                m2 = _re.search(r"^(shop|flat)\b", s, _re.I)
                if m2:
                    t = m2.group(1).title()
                # strip leading zeros
                core = _re.sub(r"^([A-Z]?)(0+)(\d)", r"\1\3", m.group(1))
                return f"{t} {core}"
            return s

        return _std_building(b), _std_floor(f), _std_flat(fl)

    bff = df.apply(infer_bld_floor_flat, axis=1, result_type="expand")
    df["__Building"] = bff[0].astype(str)
    df["__Floor"] = bff[1].astype(str)
    df["__Flat"] = bff[2].astype(str)
    # Checklist name lives in 'Eqc Type' in our extracts
    df["__Checklist"] = df.get("Eqc Type", "").astype(str)
    # Optional link
    if "URL" in df.columns:
        df["__URL"] = df["URL"].astype(str)
    else:
        df["__URL"] = ""
    return df


@app.get("/flat-report")
def flat_report_index():
    # Local test reads from Combined_EQC.csv by default; allow path override
    path = request.args.get("path") or "Combined_EQC.csv"
    if not os.path.exists(path):
        return f"EQC file not found: {path}", 404
    df = prepare_frame(robust_read_eqc(path))
    projects = sorted([p for p in df["__Project"].astype(str).str.strip().unique() if p])
    proj = request.args.get("project") or (projects[0] if projects else None)
    dfp = df[df["__Project"].astype(str).str.strip().eq(proj)] if proj else df.iloc[0:0]

    buildings_all = sorted({(b if b else "UNKNOWN") for b in dfp["__Building"].astype(str).unique()})
    # Prefer a real building over UNKNOWN by default
    default_building = next((b for b in buildings_all if str(b).upper() != "UNKNOWN"), (buildings_all[0] if buildings_all else None))
    building = request.args.get("building") or default_building

    dfl = dfp[dfp["__Building"].astype(str).fillna("").replace("", "UNKNOWN").eq(building or "")]
    # Natural sort floors (Floor 1 .. Floor 21, Parking 1..)
    def _floor_key(v: str):
        import re as _re
        s = str(v or "").strip()
        m = _re.search(r"(floor|fl|level)\s*(\d+)", s, _re.I)
        if m:
            return (0, int(m.group(2)))
        m = _re.search(r"parking\s*(\d+)", s, _re.I)
        if m:
            return (1, int(m.group(1)))
        order = {"terrace": (2, 0), "development": (3, 0), "shops": (4, 0), "unknown": (9, 0)}
        key = s.lower()
        return order.get(key, (8, 0))
    floors_set = {(f if f else "UNKNOWN") for f in dfl["__Floor"].astype(str).unique()}
    # Fallback: if floors are missing or only UNKNOWN, derive from Location L0-L4 or EQC text
    def _derive_floor_from_row(row: pd.Series) -> str:
        import re as _re
        specials = r"shear\s*wall|stair\s*case|lobby|podium|development|terrace|parking|upper\s*ground|lower\s*ground|mezzanine|basement|pod"
        for loc_field in ("Location L2", "Location L3", "Location L4", "Location L1", "Location L0"):
            sv = str(row.get(loc_field, "") or "").strip()
            if not sv:
                continue
            if _re.search(specials, sv, _re.I):
                return sv
            m = _re.search(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", sv, _re.I) or _re.search(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", sv, _re.I)
            if m:
                num = m.group(m.lastindex or 1)
                return f"Floor {num}"
        # Fallback to EQC path
        eqc = str(row.get("EQC", "") or "").strip()
        if eqc:
            # split tokens and reuse same patterns
            import re as _re2
            toks = [t.strip() for t in _re2.split(r"[>/|,;:\\-–—]+", eqc) if t.strip()]
            for sv in [eqc] + toks:
                if _re2.search(specials, sv, _re2.I):
                    return sv
                m = _re2.search(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", sv, _re2.I) or _re2.search(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", sv, _re2.I) or _re2.search(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", sv, _re2.I) or _re2.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", sv, _re2.I)
                if m:
                    num = m.group(m.lastindex or 1)
                    return f"Floor {num}"
        return "UNKNOWN"

    if not floors_set or all(str(x).upper() == "UNKNOWN" for x in floors_set):
        alt = set()
        for _, r in dfl.iterrows():
            val = _derive_floor_from_row(r)
            if val:
                alt.add(val)
        floors_set = alt if alt else floors_set
    floors_all = sorted(floors_set, key=_floor_key)
    # Prefer a numeric Floor N as the default if available, else first non-UNKNOWN
    import re as _re_df
    default_floor = next((f for f in floors_all if _re_df.search(r"\bfloor\s*\d+\b", str(f), _re_df.I)), None)
    if not default_floor:
        default_floor = next((f for f in floors_all if str(f).upper() != "UNKNOWN"), (floors_all[0] if floors_all else None))
    floor = request.args.get("floor") or default_floor

    if floor == "ALL":
        dff = dfl.copy()
    else:
        dff = dfl[dfl["__Floor"].astype(str).fillna("").replace("", "UNKNOWN").eq(floor or "")]
    # Natural sort flats (Flat/Shop 101, 101A etc.)
    def _flat_key(v: str):
        import re as _re
        s = str(v or "").strip()
        m = _re.search(r"(shop|flat|unit|apt|apartment)\s*([A-Z]?\d+)([A-Z]?)", s, _re.I)
        if m:
            num = int(_re.sub(r"\D", "", m.group(2))) if _re.sub(r"\D", "", m.group(2)) else 0
            suf = m.group(3).upper() if m.group(3) else ""
            return (0, num, suf)
        # Fallback: any number in string
        m2 = _re.search(r"(\d+)", s)
        if m2:
            return (1, int(m2.group(1)), "")
        return (9, 0, s.lower())
    flats_set = {(x if x else "UNKNOWN") for x in dff["__Flat"].astype(str).unique()}
    # Prefer only units that look like Flat/Shop for dropdown, but fall back to all when none match
    import re as _re_fl
    flats_filtered = [x for x in flats_set if _re_fl.match(r"^(?:Flat|Shop)\s+", str(x))]
    flats_all = sorted(flats_filtered if flats_filtered else flats_set, key=_flat_key)
    default_flat = next((x for x in flats_all if str(x).upper() != "UNKNOWN"), (flats_all[0] if flats_all else None))
    flat = request.args.get("flat") or ("ALL" if floor == "ALL" else default_flat)

    # Scope for selected flat
    if floor == "ALL":
        # Building-wise: use entire building as scope
        scope_flat = dff.copy()
        scope_floor = dfl.copy()
    else:
        if (flat or "").upper() == "ALL":
            scope_flat = dff.copy()
        else:
            scope_flat = dff[dff["__Flat"].astype(str).fillna("").replace("", "UNKNOWN").eq(flat or "")].copy()
        scope_floor = dfl.copy()

    # Build grouped structure: category -> list of items
    items_by_cat: Dict[Tuple[int, str], List[Dict]] = {}
    # Collect all possible checklist names across this project (for "show all sections")
    all_checklists = set(dfp["__Checklist"].astype(str).str.strip())
    def _latest_per_checklist(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame.iloc[0:0]
        tmp = frame.copy()
        tmp["__ChecklistKey"] = tmp["__Checklist"].astype(str).str.strip().str.lower()
        tmp["__DateOrd"] = pd.to_datetime(tmp["__Date"]).fillna(pd.Timestamp.min)
        idxs = tmp.sort_values("__DateOrd").groupby("__ChecklistKey").tail(1).index
        return tmp.loc[idxs]

    # When ALL floors are selected, do not populate items (UI should show only export option)
    if floor == "ALL":
        reps_flat = scope_flat.iloc[0:0]
        reps_floor = scope_floor.iloc[0:0]
    else:
        reps_flat = _latest_per_checklist(scope_flat)
        reps_floor = _latest_per_checklist(scope_floor)

    # Populate items: category 1 (RCC/Structural) from floor-level reps; others from flat-level reps
    for src, only_cat1 in ((reps_floor, True), (reps_flat, False)):
        if src is None or src.empty:
            continue
        for _, row in src.iterrows():
            cl = row["__Checklist"]
            url = row.get("__URL", "")
            stage = row.get("__Stage", "")
            status = str(row.get("Status", "")).strip()
            idx, cat = map_checklist_to_category(cl)
            if only_cat1 and idx != 0:
                continue
            if (idx, cat) not in items_by_cat:
                items_by_cat[(idx, cat)] = []
            items_by_cat[(idx, cat)].append({
                "checklist": cl,
                "url": url,
                "stage": stage,
                "status": status,
                "date": row.get("__Date"),
            })
    # Ensure all categories are present even if empty, for consistent display
    ordered_keys_all: List[Tuple[int, str]] = []
    for i, (label, _) in enumerate(CATEGORY_ORDER):
        ordered_keys_all.append((i, label))
    ordered_keys_all.append((len(CATEGORY_ORDER), UNCATEGORIZED_LABEL))

    return render_template(
        "flat_report.html",
        path=path,
        projects=projects,
        selected_project=proj,
        buildings=buildings_all,
        selected_building=building,
        floors=floors_all,
        selected_floor=floor,
        flats=flats_all,
        selected_flat=flat,
        items_by_cat=items_by_cat,
        ordered_keys_all=ordered_keys_all,
        show_only_export=(floor == "ALL"),
    )


@app.get("/flat-report/export")
def flat_report_export():
    # Build an Excel status grid for selected Project/Building/Floor
    path = request.args.get("path") or "Combined_EQC.csv"
    if not os.path.exists(path):
        return f"EQC file not found: {path}", 404
    df = prepare_frame(robust_read_eqc(path))
    projects = sorted([p for p in df["__Project"].astype(str).str.strip().unique() if p])
    proj = request.args.get("project") or (projects[0] if projects else None)
    dfp = df[df["__Project"].astype(str).str.strip().eq(proj)] if proj else df.iloc[0:0]
    building = request.args.get("building") or ""
    dfl = dfp[dfp["__Building"].astype(str).fillna("").replace("", "UNKNOWN").eq(building or "")]
    floor = request.args.get("floor") or ""
    if floor == "ALL":
        dff = dfl.copy()
    else:
        dff = dfl[dfl["__Floor"].astype(str).fillna("").replace("", "UNKNOWN").eq(floor or "")]

    # Determine flats on this floor
    # Natural-sort flats like in UI
    def _flat_key(v: str):
        import re as _re
        s = str(v or "").strip()
        m = _re.search(r"(shop|flat|unit|apt|apartment)\s*([A-Z]?\d+)([A-Z]?)", s, _re.I)
        if m:
            num = int(_re.sub(r"\D", "", m.group(2))) if _re.sub(r"\D", "", m.group(2)) else 0
            suf = m.group(3).upper() if m.group(3) else ""
            return (0, num, suf)
        m2 = _re.search(r"(\d+)", s)
        if m2:
            return (1, int(m2.group(1)), "")
        return (9, 0, s.lower())
    flats = sorted(set(dff["__Flat"].astype(str).fillna("").replace("", "UNKNOWN")), key=_flat_key)
    # Build latest rows per (flat, checklist)
    tmp = dff.copy()
    tmp["__ChecklistKey"] = tmp["__Checklist"].astype(str).str.strip().str.lower()
    tmp["__FlatKey"] = tmp["__Flat"].astype(str).fillna("").replace("", "UNKNOWN")
    tmp["__DateOrd"] = pd.to_datetime(tmp["__Date"]).fillna(pd.Timestamp.min)
    tmp = tmp.sort_values("__DateOrd").groupby(["__FlatKey", "__ChecklistKey"], as_index=False).tail(1)

    # Determine checklist columns as union across this floor (use readable names)
    checklist_cols = list(dict.fromkeys(tmp["__Checklist"].astype(str).tolist()))

    # Build status matrix
    def status_symbol(status: str) -> str:
        up = (status or "").strip().upper()
        if up == "PASSED":
            return "✓"
        if "PROGRESS" in up:
            return "–"
        return "–" if up else "–"  # unknown treated as dash

    grid = []
    for flt in flats:
        row = {"Flat": flt}
        sub = tmp[tmp["__FlatKey"].eq(flt)]
        present = set(sub["__Checklist"].astype(str))
        any_present = bool(present)
        all_complete = True
        for col in checklist_cols:
            rec = sub[sub["__Checklist"].astype(str).eq(col)]
            if rec.empty:
                row[col] = "✗"
                all_complete = False
            else:
                sym = status_symbol(str(rec.iloc[0].get("Status", "")))
                row[col] = sym
                if sym != "✓":
                    all_complete = False
        # Overall status per flat
        row["Overall"] = "✓" if all_complete and any_present else ("–" if any_present else "✗")
        grid.append(row)

    out_df = pd.DataFrame(grid, columns=["Flat", "Overall"] + checklist_cols)
    # Write Excel with basic coloring
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = f"{(building or 'Building')} - {(floor or 'Floor')}"
    # Add building name header for clarity
    ws.cell(row=1, column=1, value="Building:")
    ws.cell(row=1, column=2, value=building)
    # Leave row 2 blank, start data at row 3
    ws.append([])
    start_row = 3
    for r in dataframe_to_rows(out_df, index=False, header=True):
        ws.append(r)
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    red = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
    # Apply coloring only over data rows (excluding header at start_row)
    for row in ws.iter_rows(min_row=start_row+1, min_col=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            val = str(cell.value or "")
            if val == "✓":
                cell.fill = green
            elif val == "–":
                cell.fill = yellow
            elif val == "✗":
                cell.fill = red
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    fname = f"Flat_Status_{proj}_{building}_{(floor or 'ALL')}_{datetime.now().strftime('%Y-%m-%d')}.xlsx".replace("/", "-")
    # Persist to temp and apply borders + wrap formatting across all cells
    tmp_path = os.path.join(tempfile.gettempdir(), f"flat_export_{uuid.uuid4().hex}.xlsx")
    with open(tmp_path, "wb") as fh:
        fh.write(bio.getvalue())
    try:
        wb2 = load_workbook(tmp_path)
        thin = Side(border_style="thin", color="000000")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        align = Alignment(wrap_text=True, vertical="center")
        for ws in wb2.worksheets:
            max_row = ws.max_row or 1
            max_col = ws.max_column or 1
            for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
                for cell in row:
                    try:
                        cell.border = border
                        cell.alignment = align
                    except Exception:
                        pass
        out_bio = io.BytesIO()
        wb2.save(out_bio)
        out_bio.seek(0)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return send_file(out_bio, as_attachment=True, download_name=fname, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/")
def root_redirect():
    # Convenience: redirect base URL to the flat report page
    return redirect(url_for("flat_report_index"))


@app.get("/favicon.ico")
def favicon_empty():
    # Avoid noisy 404s for favicon during local testing
    from flask import Response
    return Response(status=204)


if __name__ == "__main__":
    # Local run only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=True)
