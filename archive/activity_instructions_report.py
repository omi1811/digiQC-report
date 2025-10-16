#!/usr/bin/env python3
"""
Activity-wise Instructions report (Weekly / Monthly / Cumulative)

Input: Combined_Instructions.csv (robust reader)
Output (in ./Activity-wise_instructions/):
  - CityLife.csv (for project 'Itrend City Life')
  - Futura.csv (for project 'FUTURA')
  - Palacio.csv (for project 'Itrend-Palacio')
  - Summary.csv (counts by Project x Activity x Stage)

Columns in per-project files:
  Sr. No., Project Name, Location / Reference, Activity, Description, Type, Stage

Derivations:
  - Stage: derived from Description + Recommendation keywords (Pre/During/Post)
  - Activity: derived from Description + Recommendation + Tag 1 keywords
  - Type: prefer Tag 1 if present, else Type L0

Rules designed to be conservative and editable. Extend ACTIVITY_RULES and STAGE_RULES
as needed. Safety is excluded (Type L0 containing 'Safety').
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------- CLI ----------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activity-wise Instructions report (Weekly/Monthly/Cumulative)")
    p.add_argument("--input", "-i", default="Combined_Instructions.csv", help="Path to Combined_Instructions file")
    p.add_argument("--mode", "-m", choices=["weekly", "monthly", "cumulative"], help="Report mode; if omitted, an interactive menu is shown")
    p.add_argument("--output-dir", "-o", default="Activity-wise_instructions", help="Output directory")
    p.add_argument("--history", default="combined_historical_data.csv", help="Path to historical Activity data (optional)")
    p.add_argument("--update-history", action="store_true", help="Append newly generated rows to the historical data (deduped)")
    return p.parse_args(argv)


# ---------------- Robust reader ----------------
def _read_instructions(path: str) -> pd.DataFrame:
    """Robust reader that tolerates a header wrapped across lines.
    Tries CSV/TSV autodetect; if it yields a single-column frame, attempts to merge header lines
    and reparse from a buffer (same approach used in build_dashboard).
    """
    # First pass attempts
    for sep in ("\t", ",", None):
        try:
            if sep is None:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine="python")
            if df.shape[1] > 1:
                # Trim column names to avoid trailing spaces (e.g., 'Tag 2 ')
                df.columns = df.columns.str.strip()
                return df
        except Exception:
            pass
    # Header-merge fallback
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        import re, io
        m = re.search(r"(?m)^\s*\d{2,},", text)
        if m:
            pos = m.start()
            header = text[:pos].replace("\r", "").replace("\n", "")
            data = text[pos:]
            new_text = header + "\n" + data
            buf = io.StringIO(new_text)
            df2 = pd.read_csv(buf, dtype=str, keep_default_na=False, engine="python")
            df2.columns = df2.columns.str.strip()
            return df2
    except Exception:
        pass
    # Last resort empty
    return pd.DataFrame()


# ---------------- Helpers ----------------
TODAY = pd.Timestamp.today().normalize()


def _parse_date_safe(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def _filter_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "Raised On Date" not in df.columns:
        return df.iloc[0:0].copy() if mode in ("weekly", "monthly") else df.copy()
    dates = df["Raised On Date"].map(_parse_date_safe)
    if mode == "weekly":
        start = TODAY.date() - pd.Timedelta(days=6).to_pytimedelta()
        mask = dates.map(lambda d: (d is not None) and (start <= d <= TODAY.date()))
        return df[mask].copy()
    if mode == "monthly":
        mon_start = TODAY.replace(day=1).date()
        mask = dates.map(lambda d: (d is not None) and (mon_start <= d <= TODAY.date()))
        return df[mask].copy()
    return df.copy()


def _build_location(row: pd.Series) -> str:
    parts = []
    for col in ("Location L0", "Location L1", "Location L2", "Location L3", "Location L4"):
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    if not parts:
        alt = row.get("Location Variable", "")
        if pd.notna(alt) and str(alt).strip():
            parts.append(str(alt).strip())
    return "/".join(parts)


# Activity and Stage rules (extendable)
ACTIVITY_RULES: List[tuple[List[str], str]] = [
    # keywords (any) -> Activity label
    (["gypsum", "bond it", "bondit"], "Gypsum"),
    (["tile", "tiling", "skirting", "granite", "dado"], "Tiling"),
    (["plaster"], "Plaster"),
    (["electrical", "switch", "db box", "db  box", "db  ", "socket", "conduit", "zari"], "Electrical work"),
    (["railing", "ms frame", "ms railing"], "MS railing"),
    (["fire fighting", "sprinkler", "sprinklar"], "Fire Fighting"),
    (["plumbing", "pipe", "grouting", "hole packing"], "Plumbing"),
    (["rcc", "shuttering", "pour", "slab", "column", "beam", "aluf", "aluform"], "RCC"),
    (["door", "frame", "hinge", "molding"], "Door fixing"),
    (["acc block", "block work", "blockwork", "aac block"], "ACC block work"),
    (["painting", "primer", "paint"], "Painting"),
    (["document", "register"], "Documentation"),
    (["material", "stacking"], "Material"),
]


def _derive_activity(desc: str, rec: str, tag1: str) -> str:
    base = " ".join([str(x or "") for x in (desc, rec, tag1)]).lower()
    for keywords, label in ACTIVITY_RULES:
        if any(k in base for k in keywords):
            return label
    # Last resort coarse bucket
    return "General"


STAGE_RULES: Dict[str, List[str]] = {
    "Pre": [
        "pre casting", "pre-concrete", "pre concrete", "pre checking", "before casting",
        "stage passing", "sequence", "pre ", "before ", "stacking", "without checklist", "planning",
    ],
    "Post": [
        "finishing", "finish", "clean", "props are removed", "after ", "post ", "debond", "loose concrete",
    ],
}


def _derive_stage(desc: str, rec: str) -> str:
    base = " ".join([str(desc or ""), str(rec or "")]).lower()
    for stage, keys in STAGE_RULES.items():
        if any(k in base for k in keys):
            return stage
    return "During"


def _get_tag1_value(row: pd.Series) -> str:
    """Return Tag-1 value from any of the common column name variants.
    Sanitizes header-like echoes such as 'Tag 1' or 'Tag L1'.
    """
    header_tokens = {"tag 1", "tag l1"}
    for col in ("Tag 1", "Tag L1", "Tag1", "Tag_1"):
        if col in row.index:
            val = str(row.get(col, "") or "").strip()
            if val and val.lower() not in header_tokens:
                return val
    return ""


# ---------- History-assisted activity classification ----------
STOPWORDS = {
    "and", "or", "the", "a", "an", "is", "are", "was", "were", "to", "of", "on", "in", "for", "by", "with",
    "at", "from", "this", "that", "it", "as", "be", "has", "have", "had", "not", "done", "provide", "provided",
}

LOCATION_TOKENS = {
    "wing", "floor", "flat", "parking", "terrace", "staircase", "lobby", "kitchen", "bedroom", "hall", "toilet",
    "wash", "basin", "balcony", "entrance", "gate", "development", "podium", "lift", "stp", "area", "common",
}


def _tokenize(text: str) -> List[str]:
    import re
    t = (text or "").lower()
    toks = re.split(r"[^a-z0-9+]+", t)
    out = []
    for tok in toks:
        tok = tok.strip()
        if not tok or len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def _load_history(path: str) -> Tuple[set, Dict[str, Dict[str, int]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Return (known_activities, activity->token->weight map, canonical name map, stage token weights).
    canonical maps lowercase activity to the most common cased name.
    stage token weights is a dict: {stage -> {token -> count}}
    """
    known: set = set()
    weights: Dict[str, Dict[str, int]] = {}
    canonical_counts: Dict[str, Dict[str, int]] = {}
    stage_weights: Dict[str, Dict[str, int]] = {"Pre": {}, "During": {}, "Post": {}}
    if not path or not os.path.exists(path):
        return known, weights, {}, stage_weights
    try:
        dfh = pd.read_csv(path, dtype=str, keep_default_na=False)
        col_act = "Activity"
        col_desc = "Description"
        col_stage = "Stage"
        if col_act not in dfh.columns or col_desc not in dfh.columns:
            return known, weights, {}, stage_weights
        for _, r in dfh.iterrows():
            act_raw = str(r.get(col_act, "")).strip()
            desc_val = str(r.get(col_desc, ""))
            stage_val = str(r.get(col_stage, "")).strip().title()
            if act_raw:
                act_l = act_raw.lower()
                known.add(act_l)
                # Canonical case counting
                canonical_counts.setdefault(act_l, {})[act_raw] = canonical_counts.setdefault(act_l, {}).get(act_raw, 0) + 1
                # Weights from description tokens
                toks = _tokenize(desc_val)
                if toks:
                    w = weights.setdefault(act_l, {})
                    for tk in toks:
                        w[tk] = w.get(tk, 0) + 1
            if stage_val in ("Pre", "During", "Post"):
                toks_s = _tokenize(desc_val)
                if toks_s:
                    sw = stage_weights.setdefault(stage_val, {})
                    for tk in toks_s:
                        sw[tk] = sw.get(tk, 0) + 1
        # Build canonical map (most frequent cased name)
        canonical: Dict[str, str] = {}
        for k, cnts in canonical_counts.items():
            canonical[k] = max(cnts.items(), key=lambda kv: kv[1])[0]
        return known, weights, canonical, stage_weights
    except Exception:
        return known, weights, {}, stage_weights


def _score_activity_by_history(text: str, weights: Dict[str, Dict[str, int]]) -> Tuple[Optional[str], int]:
    toks = _tokenize(text)
    if not toks:
        return None, 0
    best: Optional[str] = None
    best_score = 0
    for act_l, wmap in weights.items():
        s = 0
        for tk in toks:
            if tk in LOCATION_TOKENS:
                continue
            s += wmap.get(tk, 0)
        if s > best_score:
            best, best_score = act_l, s
    return best, best_score


def _score_stage_by_history(text: str, stage_weights: Dict[str, Dict[str, int]]) -> Tuple[Optional[str], int]:
    toks = _tokenize(text)
    if not toks:
        return None, 0
    best: Optional[str] = None
    best_score = 0
    for stage, wmap in stage_weights.items():
        s = 0
        for tk in toks:
            s += wmap.get(tk, 0)
        if s > best_score:
            best, best_score = stage, s
    return best, best_score


def _exclude_safety(df: pd.DataFrame) -> pd.DataFrame:
    if "Type L0" in df.columns:
        mask = ~df["Type L0"].astype(str).str.contains("safety", case=False, na=False)
        return df[mask].copy()
    return df


def _project_key_to_filename(project: str) -> Optional[str]:
    # Map project name column to desired output filename
    if project == "FUTURA":
        return "Futura.csv"
    if project == "Itrend City Life":
        return "CityLife.csv"
    if project == "Itrend-Palacio":
        return "Palacio.csv"
    return None


def _build_rows(
    df: pd.DataFrame,
    project_label: str,
    hist_ctx: Optional[Tuple[set, Dict[str, Dict[str, int]], Dict[str, str], Dict[str, Dict[str, int]]]] = None,
) -> pd.DataFrame:
    # Construct target columns
    out_rows = []
    known_set: set = set()
    weights: Dict[str, Dict[str, int]] = {}
    canonical: Dict[str, str] = {}
    stage_weights: Dict[str, Dict[str, int]] = {}
    if hist_ctx:
        known_set, weights, canonical, stage_weights = hist_ctx
    for _, row in df.iterrows():
        desc = row.get("Description", "")
        rec = row.get("Recommendation", "")
        tag1 = _get_tag1_value(row)
        # 1) Rule-based
        activity = _derive_activity(desc, rec, tag1)
        # 2) Tag1 fallback only if matches known activities (avoid location-like values)
        if activity == "General" and tag1:
            t1 = str(tag1).strip().lower()
            if t1 in known_set:
                activity = canonical.get(t1, tag1.strip())
        # 3) History-based scoring if still General
        if activity == "General" and weights:
            text = " ".join([str(x or "") for x in (desc, rec, tag1)])
            pred, score = _score_activity_by_history(text, weights)
            if pred and score >= 2:
                activity = canonical.get(pred, pred.title())
        # 4) If still General, mark as Unclassified
        if activity == "General" or not str(activity).strip():
            activity = "Unclassified"
        stage = _derive_stage(desc, rec)
        # Refine stage with history tokens if rules defaulted to During
        if stage == "During" and stage_weights:
            text = " ".join([str(x or "") for x in (desc, rec, tag1)])
            st_pred, st_score = _score_stage_by_history(text, stage_weights)
            if st_pred in ("Pre", "During", "Post") and st_score > 0:
                stage = st_pred
        type_val = str(tag1).strip() if str(tag1).strip() else str(row.get("Type L0", "")).strip()
        loc = _build_location(row)
        out_rows.append({
            "Project Name": project_label,
            "Location / Reference": loc,
            "Activity": activity,
            "Description": desc,
            "Type": type_val,
            "Stage": stage,
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        # Sort by Activity then Stage order
        try:
            out["Stage"] = pd.Categorical(out["Stage"], categories=["Pre", "During", "Post"], ordered=True)
        except Exception:
            pass
        out = out.sort_values(by=["Activity", "Stage", "Location / Reference", "Description"], kind="stable").reset_index(drop=True)
        out.insert(0, "Sr. No.", range(1, len(out) + 1))
    else:
        out = pd.DataFrame(columns=["Sr. No.", "Project Name", "Location / Reference", "Activity", "Description", "Type", "Stage"])
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    mode = args.mode
    if not mode:
        print("Select report mode:")
        print("  1) Weekly")
        print("  2) Monthly")
        print("  3) Cumulative")
        choice = input("Enter choice [1-3]: ").strip()
        mode = {"1": "weekly", "2": "monthly", "3": "cumulative"}.get(choice, "weekly")

    os.makedirs(args.output_dir, exist_ok=True)

    df = _read_instructions(args.input)
    if df is None or df.empty:
        # Write empty scaffolding files
        for name in ("CityLife.csv", "Futura.csv", "Palacio.csv", "Summary.csv"):
            pd.DataFrame(columns=["Sr. No.", "Project Name", "Location / Reference", "Activity", "Description", "Type", "Stage"]).to_csv(os.path.join(args.output_dir, name), index=False)
        print("No instructions found; wrote empty files.")
        return

    # Exclude Safety and DEMO projects
    df = _exclude_safety(df)
    if "Location L0" in df.columns:
        demo_mask = ~df["Location L0"].astype(str).str.contains("demo", case=False, na=False)
        df = df[demo_mask].copy()

    # Filter by timeframe
    df_win = _filter_by_mode(df, mode)

    # Split by project and write
    allowed = {"FUTURA", "Itrend City Life", "Itrend-Palacio"}
    all_outputs: List[pd.DataFrame] = []
    # Load history context
    hist_ctx = _load_history(args.history)
    for project, sub in df_win.groupby(df_win.get("Location L0", pd.Series(["" for _ in range(len(df_win))], index=df_win.index))):
        project = str(project).strip()
        if not project or project not in allowed:
            continue
        fname = _project_key_to_filename(project)
        if not fname:
            continue
        # Build rows
        proj_label = "Itrend Futura" if project == "FUTURA" else ("Itrend City Life" if project == "Itrend City Life" else "Itrend Palacio")
        out_df = _build_rows(sub, proj_label, hist_ctx)
        out_path = os.path.join(args.output_dir, fname)
        out_df.to_csv(out_path, index=False)
        print(f"✅ Wrote {mode} activity-wise instructions for {project} -> {out_path}")
        all_outputs.append(out_df.assign(__ProjectKey=project))

    # Summary across projects
    if all_outputs:
        cat = pd.concat(all_outputs, ignore_index=True)
        summary = (
            cat.groupby(["__ProjectKey", "Activity", "Stage"], observed=False).size().unstack(fill_value=0)
        )
        # Ensure Pre/During/Post columns present
        for col in ("Pre", "During", "Post"):
            if col not in summary.columns:
                summary[col] = 0
        summary = summary[["Pre", "During", "Post"]]
        summary["Total"] = summary.sum(axis=1)
        summary = summary.reset_index().rename(columns={"__ProjectKey": "Project"})
        # Map to friendly names
        summary["Project"] = summary["Project"].map({
            "FUTURA": "Itrend Futura",
            "Itrend City Life": "Itrend City Life",
            "Itrend-Palacio": "Itrend Palacio",
        })
        summary.to_csv(os.path.join(args.output_dir, "Summary.csv"), index=False)
        print(f"✅ Wrote summary -> {os.path.join(args.output_dir, 'Summary.csv')}")
    else:
        # Still write empty Summary
        pd.DataFrame(columns=["Project", "Activity", "Pre", "During", "Post", "Total"]).to_csv(os.path.join(args.output_dir, "Summary.csv"), index=False)
        print("No project outputs generated; wrote empty Summary.csv")

    # Optionally update history by appending newly generated rows (deduped)
    if args.update_history and all_outputs:
        hist_path = args.history or "combined_historical_data.csv"
        try:
            new_hist = pd.concat(all_outputs, ignore_index=True)[["Project Name", "Location / Reference", "Activity", "Description", "Type", "Stage"]]
            if os.path.exists(hist_path):
                old = pd.read_csv(hist_path, dtype=str, keep_default_na=False)
                merged = pd.concat([old, new_hist], ignore_index=True).drop_duplicates(subset=["Project Name", "Location / Reference", "Activity", "Description", "Type", "Stage"]).reset_index(drop=True)
            else:
                merged = new_hist.drop_duplicates().reset_index(drop=True)
            merged.to_csv(hist_path, index=False)
            print(f"✅ History updated -> {hist_path} (rows: {len(merged)})")
        except Exception as e:
            print(f"Failed to update history: {e}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
