#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import re
import tempfile
import uuid
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, send_file, redirect, url_for, session, send_from_directory
from flask import jsonify
import pandas as pd
import analysis_eqc as EQC
import analysis_issues as ISS
from project_utils import canonicalize_project_name, canonical_project_from_row
import subprocess
from openpyxl import load_workbook
from openpyxl.styles import Border, Side, Alignment

# --- Helpers (schema + stage normalization) ---

def parse_date_safe(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def normalize_stage(stage: str) -> str:
    # Delegate to analysis_eqc for consistency
    return EQC.normalize_stage(stage)


def canonical_project(row: pd.Series) -> str:
    # Use shared helper with fallback order and robust normalization
    return canonical_project_from_row(row)


def canonical_project_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized version of canonical_project for better performance on large datasets.
    
    This implementation is more efficient for large datasets by minimizing row-by-row operations.
    """
    # Initialize result series with empty strings
    result = pd.Series("", index=df.index)
    
    # Try columns in preference order, filling in empties from previous attempts
    for col in ("Location L0", "Project", "Project Name"):
        if col not in df.columns:
            continue
            
        # Get values where result is still empty
        mask_needs_value = result == ""
        if not mask_needs_value.any():
            break  # All values filled
            
        # Get and canonicalize values from this column
        series = df.loc[mask_needs_value, col].astype(str).str.strip()
        # Only canonicalize non-empty strings
        canonicalized = series.apply(lambda x: canonicalize_project_name(x) if x else "")
        
        # Update result where we found values
        result.loc[mask_needs_value] = canonicalized
    
    return result


def read_eqc_robust(fobj: io.BytesIO) -> pd.DataFrame:
    fobj.seek(0)
    last_err: Exception | None = None
    for sep in ("\t", ",", None):
        try:
            if sep is None:
                df = pd.read_csv(fobj, dtype=str, keep_default_na=False, sep=None, engine="python")
            else:
                fobj.seek(0)
                df = pd.read_csv(fobj, dtype=str, keep_default_na=False, sep=sep, engine="python")
            return df
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Failed to read uploaded file")


# --- Excel formatting helpers (borders + wrap) ---

def _apply_borders_wrap_to_wb(wb) -> None:
    """Apply thin borders and wrap text to all cells in all sheets of an openpyxl Workbook."""
    thin = Side(border_style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    align = Alignment(wrap_text=True, vertical="center")
    for ws in wb.worksheets:
        max_row = ws.max_row or 1
        max_col = ws.max_column or 1
        for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
            for cell in row:
                try:
                    cell.border = border
                    cell.alignment = align
                except Exception:
                    # Skip cells that cannot be styled (unlikely)
                    pass


def _format_xlsx_from_path(xlsx_path: str) -> io.BytesIO:
    """Load an existing .xlsx file, apply borders + wrap to all sheets, return BytesIO of the result."""
    wb = load_workbook(xlsx_path)
    _apply_borders_wrap_to_wb(wb)
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio


# --- Session-scoped CSV management ---

# Cache for file discovery to avoid repeated os.walk() calls
_file_cache: Dict[str, Tuple[float, List[str]]] = {}
_CACHE_TTL = 60  # seconds

def _find_files_cached(pattern_check, cache_key: str) -> List[str]:
    """Find files matching a pattern with caching to avoid repeated os.walk() calls."""
    import time
    current_time = time.time()
    
    # Check cache
    if cache_key in _file_cache:
        cache_time, cached_files = _file_cache[cache_key]
        if current_time - cache_time < _CACHE_TTL:
            # Validate cached files still exist
            valid_files = [f for f in cached_files if os.path.exists(f)]
            if valid_files:
                return valid_files
    
    # Perform search
    candidates: List[str] = []
    try:
        for root, _dirs, files in os.walk(os.getcwd()):
            for f in files:
                if pattern_check(f):
                    candidates.append(os.path.join(root, f))
    except Exception:
        candidates = []
    
    # Update cache
    _file_cache[cache_key] = (current_time, candidates)
    return candidates


def _ensure_upload_dir() -> str:
    root = os.path.join(tempfile.gettempdir(), "eqc_uploads")
    os.makedirs(root, exist_ok=True)
    return root


def _save_session_file(file_storage) -> tuple[str, str]:
    """Save uploaded file to a temp location and return (path, display_name)."""
    upload_dir = _ensure_upload_dir()
    ext = ".csv"
    name = getattr(file_storage, "filename", "Combined_EQC.csv") or "Combined_EQC.csv"
    try:
        base_ext = os.path.splitext(name)[1]
        if base_ext:
            ext = base_ext
    except Exception:
        pass
    uid = uuid.uuid4().hex
    out_path = os.path.join(upload_dir, f"eqc_{uid}{ext}")
    file_storage.stream.seek(0)
    with open(out_path, "wb") as out:
        out.write(file_storage.read())
    return out_path, name


def _get_session_eqc_path() -> str | None:
    p = session.get("eqc_file_path")
    if p and os.path.exists(p):
        return p
    return None


# Issues file session management
def _ensure_issues_upload_dir() -> str:
    root = os.path.join(tempfile.gettempdir(), "issues_uploads")
    os.makedirs(root, exist_ok=True)
    return root


def _save_session_issues_file(file_storage) -> tuple[str, str]:
    upload_dir = _ensure_issues_upload_dir()
    ext = ".csv"
    name = getattr(file_storage, "filename", "Combined_Instructions.csv") or "Combined_Instructions.csv"
    try:
        base_ext = os.path.splitext(name)[1]
        if base_ext:
            ext = base_ext
    except Exception:
        pass
    uid = uuid.uuid4().hex
    out_path = os.path.join(upload_dir, f"issues_{uid}{ext}")
    file_storage.stream.seek(0)
    with open(out_path, "wb") as out:
        out.write(file_storage.read())
    return out_path, name


def _get_session_issues_path() -> str | None:
    p = session.get("issues_file_path")
    if p and os.path.exists(p):
        return p
    return None


# --- EQC summaries (cumulative/month/daily) ---

def eqc_summaries(df: pd.DataFrame, target: date) -> Dict[str, Dict[str, Dict[str, int]]]:
    df = df.copy()
    # Drop known footer/summary rows like analysis_eqc._read_and_clean
    for footer_col in ["Total EQC Stages", "Fail Stages", "% Fail"]:
        if footer_col in df.columns:
            df = df[df[footer_col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "-"])]
    if "Stage" in df.columns and "Eqc Type" in df.columns:
        mask_empty = df["Stage"].astype(str).str.strip().eq("") & df["Eqc Type"].astype(str).str.strip().eq("")
        if mask_empty.any():
            df = df[~mask_empty]
    # DEMO filter
    for col in ("Project", "Project Name", "Location L0"):
        if col in df.columns:
            mask_demo = df[col].astype(str).str.contains("DEMO", case=False, na=False)
            df = df[~mask_demo]
    # Stages and dates
    if "Stage" not in df.columns:
        # If input is malformed, return empty summary
        return {}
    df["__Stage"] = df["Stage"].map(EQC.normalize_stage)
    dates = df.get("Date", pd.Series([None] * len(df), index=df.index)).astype(str).map(EQC._parse_date_safe)
    # Projects - use vectorized version for better performance
    df["__ProjectKey"] = canonical_project_vectorized(df)
    projects = [p for p in sorted(df["__ProjectKey"].astype(str).str.strip().unique()) if p]

    def counts_daily_post_plus_other(frame: pd.DataFrame) -> Dict[str, int]:
        # Raw counts with Post including Other; no cumulative roll-up here
        if frame is None or frame.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        c = frame.groupby("__Stage", dropna=False)["__Stage"].count()
        n_pre = int(c.get("Pre", 0)); n_during = int(c.get("During", 0)); n_post = int(c.get("Post", 0)); n_other = int(c.get("Other", 0))
        return {"Pre": n_pre, "During": n_during, "Post": n_post + n_other}

    def counts_raw(frame: pd.DataFrame) -> Dict[str, int]:
        # Raw counts without any cumulative roll-up and without adding Other
        if frame is None or frame.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        c = frame.groupby("__Stage", dropna=False)["__Stage"].count()
        n_pre = int(c.get("Pre", 0)); n_during = int(c.get("During", 0)); n_post = int(c.get("Post", 0))
        return {"Pre": n_pre, "During": n_during, "Post": n_post}

    def counts_cumulative_weeklylogic(frame: pd.DataFrame) -> Dict[str, int]:
        """Cumulative roll-up using the same mapping as the console (weekly report) and workbook:

        - Pre = total rows
        - During = During + Post + Other
        - Post = Post + Other

        Where 'Other' are rows whose Stage is not pre/during/post/reinforce/shutter.
        """
        if frame is None or frame.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        s = frame.get("Stage", pd.Series(["" ] * len(frame), index=frame.index)).astype(str).str.lower()
        pre = s.str.contains("pre", na=False)
        during = s.str.contains("during", na=False)
        post = s.str.contains("post", na=False)
        reinf = s.str.contains("reinforce", na=False)
        shut = s.str.contains("shutter", na=False)
        other = ~(pre | during | post | reinf | shut)
        total = int(len(s))
        d_count = int(during.sum())
        p_count = int(post.sum())
        o_count = int(other.sum())
        return {"Pre": total, "During": d_count + p_count + o_count, "Post": p_count + o_count}

    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for proj in projects:
        sub = df[df["__ProjectKey"].astype(str).str.strip() == proj]
        dates_sub = dates.loc[sub.index]
        # Vectorized date comparisons for better performance
        month_mask = (dates_sub.dt.year == target.year) & (dates_sub.dt.month == target.month) if hasattr(dates_sub, 'dt') else dates_sub.map(lambda d: bool(d and d.year == target.year and d.month == target.month))
        today_mask = dates_sub == target
        out[proj] = {
            # All-time: cumulative roll-up to match workbook 'Cumulative' ALL sums (weekly report mapping)
            "all": counts_cumulative_weeklylogic(sub),
            # This month: raw counts with Post += Other
            "month": counts_daily_post_plus_other(sub[month_mask]),
            # Today: raw counts with Post += Other
            "today": counts_daily_post_plus_other(sub[today_mask]),
        }
    return out


# --- Flask app ---
app = Flask(__name__)
# Simple secret key for sessions (override via env SECRET_KEY in production)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-please")

# --- AdSense config (optional) ---
app.config["ADSENSE_CLIENT_ID"] = (os.environ.get("ADSENSE_CLIENT_ID", "").strip())
app.config["ADSENSE_ENABLE"] = (os.environ.get("ADSENSE_ENABLE", "1").strip().lower() in {"1","true","yes","on"}) if app.config["ADSENSE_CLIENT_ID"] else False

@app.context_processor
def _inject_adsense_config():
    return {
        "ADSENSE_CLIENT_ID": app.config.get("ADSENSE_CLIENT_ID", ""),
        "ADSENSE_ENABLE": app.config.get("ADSENSE_ENABLE", False),
    }

# Hardcoded simple credentials (no database)
AUTH_USERNAME = os.environ.get("APP_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("APP_PASSWORD", "admin123")

# Default external person name for issues (can be overridden via UI). Use empty by default.
DEFAULT_ISSUES_EXTERNAL = os.environ.get("ISSUES_EXTERNAL_NAME", "")

# Public route for ads.txt at domain root
@app.get("/ads.txt")
def ads_txt():
    # Serve the static/ads.txt file as plain text without requiring login
    return send_from_directory(os.path.join(app.root_path, "static"), "ads.txt", mimetype="text/plain")

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


@app.post("/set-eqc-file")
@login_required
def set_eqc_file():
    f = request.files.get("eqc")
    nxt = request.args.get("next") or request.referrer or url_for("index")
    if not f:
        return redirect(nxt)
    try:
        path, disp = _save_session_file(f)
    except Exception as e:
        return f"Failed to save file: {e}", 400
    session["eqc_file_path"] = path
    session["eqc_file_name"] = disp
    return redirect(nxt)


@app.post("/set-issues-file")
@login_required
def set_issues_file():
    f = request.files.get("issues")
    nxt = request.args.get("next") or request.referrer or url_for("index")
    if not f:
        return redirect(nxt)
    try:
        path, disp = _save_session_issues_file(f)
    except Exception as e:
        return f"Failed to save issues file: {e}", 400
    session["issues_file_path"] = path
    session["issues_file_name"] = disp
    return redirect(nxt)


@app.post("/set-issues-external")
@login_required
def set_issues_external():
    name = (request.form.get("external_name") or "").strip()
    nxt = request.args.get("next") or request.referrer or url_for("index")
    # If blank: set to empty to indicate "no external auditor"
    session["issues_external_name"] = name
    return redirect(nxt)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            session["user"] = username
            # After login, open uploads modal by default to set session files
            return redirect(url_for("index", openUpload=1))
        return render_template("login.html", error="Invalid credentials")
    if session.get("user"):
        return redirect(url_for("index"))
    return render_template("login.html")

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
        # Optional helper: suggest common "Raised By" names if Issues file is available
        top_names: List[Tuple[str, int]] = []
        path = _get_session_issues_path()
        if not path or not os.path.exists(path):
            # Autodetect most recent Instruction CSV in workspace using cached search
            def check_instruction(f: str) -> bool:
                fl = f.lower()
                return fl.endswith('.csv') and ('instruction' in fl or fl.startswith('csv-instruction-latest-report'))
            
            candidates = _find_files_cached(check_instruction, "instructions_csv")
            if candidates:
                path = max(candidates, key=lambda p: os.path.getmtime(p))
        try:
            if path and os.path.exists(path):
                df_issues = _robust_read_issues_path(path)
                if "Raised By" in df_issues.columns:
                    ser = df_issues["Raised By"].astype(str).str.strip()
                    ser = ser[ser != ""]
                    top = ser.value_counts().head(8)
                    top_names = [(str(idx), int(cnt)) for idx, cnt in top.items()]
        except Exception:
            top_names = []
        return render_template("index.html", top_names=top_names)


@app.get("/api/suggest-raised-by")
@login_required
def api_suggest_raised_by():
    """Return a JSON list of suggested 'Raised By' names matching the query.
    Query params: q (string), limit (int, default 8)
    """
    q = (request.args.get("q") or "").strip()
    try:
        limit = int(request.args.get("limit", 8))
    except Exception:
        limit = 8
    path = _get_session_issues_path()
    if not path or not os.path.exists(path):
        return jsonify([])
    try:
        df = _robust_read_issues_path(path)
        if "Raised By" not in df.columns:
            return jsonify([])
        ser = df["Raised By"].astype(str).str.strip()
        ser = ser[ser != ""]
        if q:
            ser = ser[ser.str.contains(q, case=False, na=False)]
        counts = ser.value_counts()
        out = [{"name": str(idx), "count": int(cnt)} for idx, cnt in counts.head(limit).items()]
        return jsonify(out)
    except Exception:
        return jsonify([])


@app.get("/reports")
@login_required
def reports_page():
    return render_template("reports.html")


@app.get("/dashboards")
@login_required
def dashboards_page():
    return render_template("dashboards.html")


@app.post("/daily-dashboard")
@login_required
def daily_dashboard():
    f = request.files.get("eqc")
    if f:
        buf = io.BytesIO(f.read())
    else:
        p = _get_session_eqc_path() or os.path.join(os.getcwd(), "Combined_EQC.csv")
        if not p or not os.path.exists(p):
            # As a last resort, try to autodetect a likely EQC CSV using cached search
            def check_eqc(fn: str) -> bool:
                fl = fn.lower()
                return fl.endswith('.csv') and (fl.startswith('combined_eqc') or 'eqc' in fl)
            
            candidates = _find_files_cached(check_eqc, "eqc_csv")
            if candidates:
                p = max(candidates, key=lambda q: os.path.getmtime(q))
            else:
                return redirect(url_for("index"))
        try:
            with open(p, "rb") as fh:
                buf = io.BytesIO(fh.read())
        except Exception as e:
            return f"Failed to read stored file: {e}", 400
    try:
        df = read_eqc_robust(buf)
    except Exception as e:
        return f"Failed to read file: {e}", 400
    today = date.today()
    data = eqc_summaries(df, today)
    # Sort projects for stable display
    data_sorted = {k: data[k] for k in sorted(data.keys())}
    return render_template("dashboard.html", data=data_sorted, target=today.strftime("%d-%m-%Y"))


@app.get("/daily-dashboard")
@login_required
def daily_dashboard_get():
    # Support GET to render using session EQC file or sensible fallbacks
    p = _get_session_eqc_path() or os.path.join(os.getcwd(), "Combined_EQC.csv")
    if not p or not os.path.exists(p):
        # Autodetect most recent EQC-like CSV using cached search
        def check_eqc(fn: str) -> bool:
            fl = fn.lower()
            return fl.endswith('.csv') and (fl.startswith('combined_eqc') or 'eqc' in fl)
        
        candidates = _find_files_cached(check_eqc, "eqc_csv")
        if candidates:
            p = max(candidates, key=lambda q: os.path.getmtime(q))
        else:
            # Nothing to show; fallback to index
            return redirect(url_for("index"))
    try:
        with open(p, "rb") as fh:
            buf = io.BytesIO(fh.read())
        df = read_eqc_robust(buf)
    except Exception as e:
        return f"Failed to read EQC file: {e}", 400
    today = date.today()
    data = eqc_summaries(df, today)
    data_sorted = {k: data[k] for k in sorted(data.keys())}
    return render_template("dashboard.html", data=data_sorted, target=today.strftime("%d-%m-%Y"))


@app.post("/weekly-report")
@login_required
def weekly_report():
    f = request.files.get("eqc")
    if f:
        # Use a temporary directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "Combined_EQC.csv")
            with open(in_path, "wb") as out:
                out.write(f.read())
            import subprocess, sys as _sys
            cmd = [_sys.executable, "Weekly_report.py", "--input", in_path]
            proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
            if proc.returncode != 0:
                return f"Failed to build weekly workbook.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}", 500
            out_xlsx = os.path.join(os.getcwd(), "EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx")
            if not os.path.exists(out_xlsx):
                return f"Workbook not found after generation.", 500
            # Apply formatting before sending
            content = _format_xlsx_from_path(out_xlsx)
            try:
                os.remove(out_xlsx)
            except Exception:
                pass
            today_str = date.today().strftime("%d-%m-%Y")
            dl_name = f"EQC_Weekly_Monthly_Cumulative_AllProjects_{today_str}.xlsx"
            return send_file(content, as_attachment=True, download_name=dl_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        p = _get_session_eqc_path()
        if not p:
            return redirect(url_for("index"))
        import subprocess, sys as _sys
        cmd = [_sys.executable, "Weekly_report.py", "--input", p]
        proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        if proc.returncode != 0:
            return f"Failed to build weekly workbook.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}", 500
        out_xlsx = os.path.join(os.getcwd(), "EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx")
        if not os.path.exists(out_xlsx):
            return f"Workbook not found after generation.", 500
        # Apply formatting before sending
        content = _format_xlsx_from_path(out_xlsx)
        try:
            os.remove(out_xlsx)
        except Exception:
            pass
        today_str = date.today().strftime("%d-%m-%Y")
        dl_name = f"EQC_Weekly_Monthly_Cumulative_AllProjects_{today_str}.xlsx"
        return send_file(content, as_attachment=True, download_name=dl_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/weekly-report")
@login_required
def weekly_report_get():
    # Avoid 405 if someone opens this URL directly; send them to the form
    return redirect(url_for("index"))


# ---------------- Flat-wise report (Floor/Flat) ----------------

# Category order and mapping adapted from flat_app.py
from typing import Tuple as _Tuple, List as _List

CATEGORY_ORDER: _List[_Tuple[str, _List[str]]] = [
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


def _map_checklist_to_category(name: str) -> _Tuple[int, str]:
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


def _robust_read_eqc_path(path: str) -> pd.DataFrame:
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


def _robust_read_issues_path(path: str) -> pd.DataFrame:
    # Try normal read via ISS helper first
    try:
        df = ISS._read_instructions(path)
    except Exception:
        df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and df.shape[1] > 1:
        return df
    # Fallback: attempt header merge if everything got jammed into one column
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        import re as _re
        m = _re.search(r"(?m)^\s*\d{2,},", text)
        if m:
            pos = m.start()
            header = text[:pos].replace("\r", "").replace("\n", "")
            data = text[pos:]
            new_text = header + "\n" + data
            buf = io.StringIO(new_text)
            df2 = pd.read_csv(buf, dtype=str, keep_default_na=False, engine='python')
            if df2.shape[1] > 1:
                return df2
    except Exception:
        pass
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Copy & filter DEMO
    df = df.copy()
    cols = [c for c in ("Project", "Project Name", "Location L0") if c in df.columns]
    if cols:
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= df[c].astype(str).str.contains("DEMO", case=False, na=False)
        df = df[~mask]
    # Stage & date
    if "Stage" in df.columns:
        df["__Stage"] = df["Stage"].map(EQC.normalize_stage)
    if "Date" in df.columns:
        df["__Date"] = df["Date"].astype(str).map(EQC._parse_date_safe)
    else:
        df["__Date"] = None
    # Project key
    df["__Project"] = df.apply(canonical_project_from_row, axis=1)

    # Preferred building label per (Project, Letter)
    import re as _re_map
    bldg_pref_map: Dict[_Tuple[str, str], str] = {}
    pref_counts: Dict[_Tuple[str, str, str], int] = {}
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
                letter = g[1] if g[0].lower() in ("wing","tower","building","block","bldg","blk") else g[0]
                key = (proj, str(letter).upper())
                pref_counts[(proj, str(letter).upper(), l1)] = pref_counts.get((proj, str(letter).upper(), l1), 0) + 1
        for (proj, letter, label), cnt in pref_counts.items():
            cur = bldg_pref_map.get((proj, letter))
            if cur is None or cnt > pref_counts.get((proj, letter, cur), 0):
                bldg_pref_map[(proj, letter)] = label

    def _infer_bff(row: pd.Series) -> _Tuple[str, str, str]:
        import re as _re
        b = str(row.get("Location L1", "") or "").strip()
        f = str(row.get("Location L2", "") or "").strip()
        fl = str(row.get("Location L3", "") or "").strip()
        sources = []
        for c in ("EQC", "Eqc", "Eqc Type", "Stage", "Status"):
            if c in row and str(row[c]).strip():
                sources.append(str(row[c]))
        for c in row.index:
            sc = str(c).strip().lower()
            if sc in {"eqc path", "eqc location", "eqc name"} and str(row[c]).strip():
                sources.append(str(row[c]))
        tokens: _List[str] = []
        for s in sources:
            tokens.extend([t.strip() for t in _re.split(r"[>/|,;:\-–—]+", str(s)) if t.strip()])
        if not b:
            patt_building = [
                _re.compile(r"\b(wing|tower|building|block)[\s\-]*([A-Za-z0-9])\b", _re.I),
                _re.compile(r"\b([A-Za-z0-9])[\s\-]*(wing|tower|building|block)\b", _re.I),
                _re.compile(r"\b(bldg|blk)\.?[\s\-]*([A-Za-z0-9])\b", _re.I),
            ]
            proj = str(row.get("__Project", "") or "").strip()
            letter_found = None
            type_found = None
            for s in sources + tokens:
                for pat in patt_building:
                    m = pat.search(s)
                    if m:
                        g1, g2 = m.groups()
                        type_token = g1 if g1.lower() in ("wing","tower","building","block","bldg","blk") else g2
                        letter_token = g2 if type_token == g1 else g1
                        letter_found = str(letter_token).strip().upper()
                        type_found = str(type_token).strip().lower()
                        break
                if b:
                    break
            if not b and letter_found:
                pref = bldg_pref_map.get((proj, letter_found))
                if pref:
                    b = pref
                elif type_found:
                    type_map = {"bldg": "Building", "blk": "Block"}
                    t = type_map.get(type_found, type_found).title()
                    b = f"{t} {letter_found}"
        if not f:
            for loc_field in ("Location L2", "Location L3", "Location L4", "Location L1", "Location L0"):
                sv = str(row.get(loc_field, "") or "").strip()
                if not sv:
                    continue
                if _re.search(r"\bshear\s*wall\b|stair\s*case|lobby|podium|development|terrace|parking", sv, _re.I):
                    f = sv
                    break
                m = _re.search(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", sv, _re.I) or _re.search(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", sv, _re.I)
                if m:
                    num = m.group(m.lastindex or 1)
                    f = f"Floor {num}"
                    break
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
                        num = m.group(m.lastindex or 1)
                        f = f"Floor {num}"
                        break
                if f:
                    break
        if not fl:
            # Require an explicit type like Flat/Shop/Unit/Apt/Room to avoid misreading product codes (e.g., "Samshield XL 1500") as flats
            patt_flat = [
                _re.compile(r"\b(flat|unit|apt|apartment|shop)[\s\-]*([A-Z]?\d{1,4}[A-Z]?)\b", _re.I),
                _re.compile(r"\b(room)[\s\-]*(\d{1,4}[A-Z]?)\b", _re.I),
            ]
            for s in sources + tokens:
                for pat in patt_flat:
                    m = pat.search(s)
                    if m:
                        val = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                        core = _re.sub(r"^([A-Z]?)(0+)(\d)", r"\1\3", val)
                        typ = m.group(1).strip().title() if (m.lastindex and m.lastindex >= 2) else "Flat"
                        if typ.lower() not in ("flat","unit","apt","apartment","shop"):
                            typ = "Flat"
                        if typ.lower() in ("unit","apt","apartment"):
                            typ = "Flat"
                        fl = f"{typ} {core}"
                        break
                if fl:
                    break
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
                t = "Flat"
                m2 = _re.search(r"^(shop|flat)\b", s, _re.I)
                if m2:
                    t = m2.group(1).title()
                core = _re.sub(r"^([A-Z]?)(0+)(\d)", r"\1\3", m.group(1))
                return f"{t} {core}"
            return s
        return (b or "").strip(), _std_floor(f), _std_flat(fl)

    bff = df.apply(_infer_bff, axis=1, result_type="expand")
    df["__Building"] = bff[0].astype(str)
    df["__Floor"] = bff[1].astype(str)
    df["__Flat"] = bff[2].astype(str)
    df["__Checklist"] = df.get("Eqc Type", "").astype(str)
    df["__URL"] = df.get("URL", "").astype(str) if "URL" in df.columns else ""
    return df


@app.get("/flat-report")
@login_required
def flat_report_index():
    # Default to session CSV if present; allow override via ?path=
    path = request.args.get("path") or _get_session_eqc_path() or "Combined_EQC.csv"
    if not os.path.exists(path):
        return f"EQC file not found: {path}", 404
    df = _prepare_frame(_robust_read_eqc_path(path))
    projects = sorted([p for p in df["__Project"].astype(str).str.strip().unique() if p])
    proj = request.args.get("project") or (projects[0] if projects else None)
    dfp = df[df["__Project"].astype(str).str.strip().eq(proj)] if proj else df.iloc[0:0]

    buildings_all = sorted({(b if b else "UNKNOWN") for b in dfp["__Building"].astype(str).unique()})
    default_building = next((b for b in buildings_all if str(b).upper() != "UNKNOWN"), (buildings_all[0] if buildings_all else None))
    building = request.args.get("building") or default_building

    dfl = dfp[dfp["__Building"].astype(str).fillna("").replace("", "UNKNOWN").eq(building or "")]

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
    if not floors_set or all(str(x).upper() == "UNKNOWN" for x in floors_set):
        alt = set()
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
            eqc = str(row.get("EQC", "") or "").strip()
            if eqc:
                toks = [t.strip() for t in _re.split(r"[>/|,;:\\-–—]+", eqc) if t.strip()]
                for sv in [eqc] + toks:
                    if _re.search(specials, sv, _re.I):
                        return sv
                    m = _re.search(r"\b(floor|fl|level)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})[\s\-]*(floor|fl|level)\b", sv, _re.I) or _re.search(r"\b(?:f|fl|lvl|l)[\s\-]*(\d{1,2})\b", sv, _re.I) or _re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:floor)?\b", sv, _re.I)
                    if m:
                        num = m.group(m.lastindex or 1)
                        return f"Floor {num}"
            return "UNKNOWN"
        for _, r in dfl.iterrows():
            val = _derive_floor_from_row(r)
            if val:
                alt.add(val)
        floors_set = alt if alt else floors_set
    floors_all = sorted(floors_set, key=_floor_key)
    import re as _re_df
    default_floor = next((f for f in floors_all if _re_df.search(r"\bfloor\s*\d+\b", str(f), _re_df.I)), None)
    if not default_floor:
        default_floor = next((f for f in floors_all if str(f).upper() != "UNKNOWN"), (floors_all[0] if floors_all else None))
    floor = request.args.get("floor") or default_floor

    if floor == "ALL":
        dff = dfl.copy()
    else:
        dff = dfl[dfl["__Floor"].astype(str).fillna("").replace("", "UNKNOWN").eq(floor or "")]

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

    flats_set = {(x if x else "UNKNOWN") for x in dff["__Flat"].astype(str).unique()}
    import re as _re_fl
    flats_filtered = [x for x in flats_set if _re_fl.match(r"^(?:Flat|Shop)\s+", str(x))]
    flats_all = sorted(flats_filtered if flats_filtered else flats_set, key=_flat_key)
    default_flat = next((x for x in flats_all if str(x).upper() != "UNKNOWN"), (flats_all[0] if flats_all else None))
    flat = request.args.get("flat") or ("ALL" if floor == "ALL" else default_flat)

    if floor == "ALL":
        scope_flat = dff.copy()
        scope_floor = dfl.copy()
    else:
        if (flat or "").upper() == "ALL":
            scope_flat = dff.copy()
        else:
            scope_flat = dff[dff["__Flat"].astype(str).fillna("").replace("", "UNKNOWN").eq(flat or "")].copy()
        scope_floor = dfl.copy()

    items_by_cat: Dict[_Tuple[int, str], _List[Dict]] = {}

    def _latest_per_checklist(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame.iloc[0:0]
        tmp = frame.copy()
        tmp["__ChecklistKey"] = tmp["__Checklist"].astype(str).str.strip().str.lower()
        tmp["__DateOrd"] = pd.to_datetime(tmp["__Date"]).fillna(pd.Timestamp.min)
        idxs = tmp.sort_values("__DateOrd").groupby("__ChecklistKey").tail(1).index
        return tmp.loc[idxs]

    if floor == "ALL":
        reps_flat = scope_flat.iloc[0:0]
        reps_floor = scope_floor.iloc[0:0]
    else:
        reps_flat = _latest_per_checklist(scope_flat)
        reps_floor = _latest_per_checklist(scope_floor)

    for src, only_cat1 in ((reps_floor, True), (reps_flat, False)):
        if src is None or src.empty:
            continue
        for _, row in src.iterrows():
            cl = row["__Checklist"]
            url = row.get("__URL", "")
            stage = row.get("__Stage", "")
            status = str(row.get("Status", "")).strip()
            idx, cat = _map_checklist_to_category(cl)
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

    ordered_keys_all: _List[_Tuple[int, str]] = []
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
@login_required
def flat_report_export():
    path = request.args.get("path") or _get_session_eqc_path() or "Combined_EQC.csv"
    if not os.path.exists(path):
        return f"EQC file not found: {path}", 404
    df = _prepare_frame(_robust_read_eqc_path(path))
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
    tmp = dff.copy()
    tmp["__ChecklistKey"] = tmp["__Checklist"].astype(str).str.strip().str.lower()
    tmp["__FlatKey"] = tmp["__Flat"].astype(str).fillna("").replace("", "UNKNOWN")
    tmp["__DateOrd"] = pd.to_datetime(tmp["__Date"]).fillna(pd.Timestamp.min)
    tmp = tmp.sort_values("__DateOrd").groupby(["__FlatKey", "__ChecklistKey"], as_index=False).tail(1)

    checklist_cols = list(dict.fromkeys(tmp["__Checklist"].astype(str).tolist()))

    def status_symbol(status: str) -> str:
        up = (status or "").strip().upper()
        if up == "PASSED":
            return "✓"
        if "PROGRESS" in up:
            return "–"
        return "–" if up else "–"

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
        row["Overall"] = "✓" if all_complete and any_present else ("–" if any_present else "✗")
        grid.append(row)

    out_df = pd.DataFrame(grid, columns=["Flat", "Overall"] + checklist_cols)
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = f"{(building or 'Building')} - {(floor or 'Floor')}"
    ws.cell(row=1, column=1, value="Building:")
    ws.cell(row=1, column=2, value=building)
    ws.append([])
    start_row = 3
    for r in dataframe_to_rows(out_df, index=False, header=True):
        ws.append(r)
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    red = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
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
    # Persist to temp to reuse the formatting helper, then send formatted workbook
    tmp_path = os.path.join(tempfile.gettempdir(), f"flat_export_{uuid.uuid4().hex}.xlsx")
    with open(tmp_path, "wb") as fh:
        fh.write(bio.getvalue())
    formatted = _format_xlsx_from_path(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return send_file(formatted, as_attachment=True, download_name=fname, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------- Building-wise cumulative dashboard ----------------

@app.get("/building-dashboard")
@login_required
def building_dashboard():
    # Default to session CSV if present; allow override via ?path=
    path = request.args.get("path") or _get_session_eqc_path() or "Combined_EQC.csv"
    if not os.path.exists(path):
        return f"EQC file not found: {path}", 404
    # Reuse robust parsing and inference for project and building
    df = _prepare_frame(_robust_read_eqc_path(path))
    # Compute raw counts per normalized stage using __Stage from _prepare_frame (original/correct logic)
    def counts_by_stage(frame: pd.DataFrame) -> Dict[str, int]:
        if frame is None or frame.empty:
            return {"Pre": 0, "During": 0, "Post": 0}
        s = frame.get("__Stage", pd.Series(["" ] * len(frame), index=frame.index)).astype(str)
        c = s.value_counts()
        return {
            "Pre": int(c.get("Pre", 0)),
            "During": int(c.get("During", 0)),
            "Post": int(c.get("Post", 0)),
        }

    data: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not df.empty:
        for proj in sorted(df["__Project"].astype(str).str.strip().unique()):
            sub_p = df[df["__Project"].astype(str).str.strip().eq(proj)]
            # Prefer excluding UNKNOWN building from charts if there are known ones
            buildings = sorted([b for b in sub_p["__Building"].astype(str).replace("", "UNKNOWN").unique()])
            has_known = any(str(b).upper() != "UNKNOWN" for b in buildings)
            if has_known:
                buildings = [b for b in buildings if str(b).upper() != "UNKNOWN"]
            proj_map: Dict[str, Dict[str, int]] = {}
            for b in buildings:
                sub_b = sub_p[sub_p["__Building"].astype(str).fillna("").replace("", "UNKNOWN").eq(b)]
                proj_map[b] = counts_by_stage(sub_b)
            data[proj] = proj_map

    # Prepare a lightweight structure for Chart.js
    # { project: { labels: [buildings], pre: [], during: [], post: [] } }
    chart_data: Dict[str, Dict[str, List[int] | List[str]]] = {}
    for proj, bmap in data.items():
        labels = list(bmap.keys())
        pre = [bmap[b]["Pre"] for b in labels]
        dur = [bmap[b]["During"] for b in labels]
        post = [bmap[b]["Post"] for b in labels]
        chart_data[proj] = {"labels": labels, "pre": pre, "during": dur, "post": post}

    return render_template("building_dashboard.html", chart_data=chart_data, path=path)


# ---------------- Building-wise Issues dashboard ----------------

@app.get("/issues-building-dashboard")
@login_required
def issues_building_dashboard():
    # Determine issues file: ?path=, session, or autodetect
    path = request.args.get("path") or _get_session_issues_path()
    if not path or not os.path.exists(path):
        # Autodetect most recent Instruction CSV in workspace using cached search
        def check_instruction(f: str) -> bool:
            fl = f.lower()
            return fl.endswith('.csv') and ('instruction' in fl or fl.startswith('csv-instruction-latest-report'))
        
        candidates = _find_files_cached(check_instruction, "instructions_csv")
        if candidates:
            path = max(candidates, key=lambda p: os.path.getmtime(p))
        else:
            path = ""
    if not path or not os.path.exists(path):
        return render_template("issues_building_dashboard.html", chart_data={}, path=""), 200

    # Read and filter issues
    df = _robust_read_issues_path(path)
    # DEMO filter
    for col in ("Project", "Project Name", "Location L0"):
        if col in df.columns:
            mask_demo = df[col].astype(str).str.contains("DEMO", case=False, na=False)
            df = df[~mask_demo]
    # Quality filter (exclude Safety)
    df = ISS._filter_quality(df)

    # Reuse frame prep to infer project and building where possible
    try:
        df = _prepare_frame(df)
    except Exception:
        # Minimal fallback: project/building from L0/L1
        df = df.copy()
        df["__Project"] = df.get("Location L0", df.get("Project Name", "")).astype(str)
        df["__Building"] = df.get("Location L1", "").astype(str)

    # Optional date range filter via query params (start/end)
    # Parse dates from "Raised On Date" using ISS._parse_date_safe
    dates = df.get("Raised On Date")
    if dates is not None:
        dates = dates.map(ISS._parse_date_safe)
    else:
        dates = pd.Series([None] * len(df), index=df.index)
    start_q = request.args.get("start", "").strip()
    end_q = request.args.get("end", "").strip()
    start_d = None
    end_d = None
    if start_q:
        # accept ISO (YYYY-MM-DD) or dd-mm-YYYY (handled by parser)
        start_d = ISS._parse_date_safe(start_q)
    if end_q:
        end_d = ISS._parse_date_safe(end_q)
    if start_d or end_d:
        mask = pd.Series([True] * len(df), index=df.index)
        if start_d:
            mask &= dates.map(lambda d: (d is not None) and (d >= start_d))
        if end_d:
            mask &= dates.map(lambda d: (d is not None) and (d <= end_d))
        df = df[mask]

    # Tag External/Internal per user-configured name
    ext_name = (session.get("issues_external_name", DEFAULT_ISSUES_EXTERNAL) or "").strip()
    has_external = bool(ext_name) and ext_name.strip().lower() != "none"
    df = ISS._tag_external(df, ext_name)

    # Count issues per (project, building, category): Total/Open/Closed
    def _counts(frame: pd.DataFrame) -> Dict[str, int]:
        c = ISS._count_frame(frame)
        return {"Total": int(c.total), "Open": int(c.open), "Closed": int(c.closed)}

    data: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not df.empty:
        for proj in sorted([p for p in df["__Project"].astype(str).str.strip().unique() if p]):
            sub_p = df[df["__Project"].astype(str).str.strip().eq(proj)]
            buildings = sorted([b for b in sub_p["__Building"].astype(str).replace("", "UNKNOWN").unique()])
            has_known = any(str(b).upper() != "UNKNOWN" for b in buildings)
            if has_known:
                buildings = [b for b in buildings if str(b).upper() != "UNKNOWN"]
            proj_map: Dict[str, Dict[str, Dict[str, int]]] = {}
            for b in buildings:
                sub_b = sub_p[sub_p["__Building"].astype(str).fillna("").replace("", "UNKNOWN").eq(b)]
                cat_map: Dict[str, Dict[str, int]] = {}
                for cat, g in sub_b.groupby("__Category"):
                    cat_map[str(cat)] = _counts(g)
                # Ensure both categories present
                for missing in ("External", "Internal"):
                    if missing not in cat_map:
                        cat_map[missing] = {"Total": 0, "Open": 0, "Closed": 0}
                proj_map[b] = cat_map
            data[proj] = proj_map

    # Chart payload per project: labels, total, open, closed
    chart_data: Dict[str, Dict[str, List[int] | List[str]]] = {}
    for proj, bmap in data.items():
        labels = [b for b in bmap.keys() if str(b).strip()]
        labels.sort()
        tot_ext = [bmap[b]["External"]["Total"] for b in labels]
        open_ext = [bmap[b]["External"]["Open"] for b in labels]
        closed_ext = [bmap[b]["External"]["Closed"] for b in labels]
        tot_int = [bmap[b]["Internal"]["Total"] for b in labels]
        open_int = [bmap[b]["Internal"]["Open"] for b in labels]
        closed_int = [bmap[b]["Internal"]["Closed"] for b in labels]
        chart_data[proj] = {
            "labels": labels,
            "ext_total": tot_ext, "ext_open": open_ext, "ext_closed": closed_ext,
            "int_total": tot_int, "int_open": open_int, "int_closed": closed_int,
        }

    # Preserve ISO strings for date inputs
    start_iso = start_d.isoformat() if start_d else ""
    end_iso = end_d.isoformat() if end_d else ""

    return render_template("issues_building_dashboard.html", chart_data=chart_data, path=path, start=start_iso, end=end_iso, external_name=ext_name, has_external=has_external)


# ---------------- Issues Daily Dashboard (per project) ----------------

def _issues_summary_by_project(df: pd.DataFrame, target: date) -> Dict[str, Dict[str, ISS.Counts]]:
    """Aggregate Quality issues (Safety excluded) by project for today/month/all.
    Returns { project: { 'today': Counts, 'month': Counts, 'all': Counts } }"""
    out: Dict[str, Dict[str, ISS.Counts]] = {}
    if df is None or df.empty:
        return out
    df = df.copy()
    # DEMO filter
    for col in ("Project", "Project Name", "Location L0"):
        if col in df.columns:
            mask_demo = df[col].astype(str).str.contains("DEMO", case=False, na=False)
            df = df[~mask_demo]
    # Quality-only
    df = ISS._filter_quality(df)
    # Canonical project key
    df["__ProjectKey"] = df.apply(canonical_project_from_row, axis=1)
    # Dates
    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)
    masks = ISS._timeframe_masks(dates, target)
    for proj, sub in df.groupby("__ProjectKey"):
        res: Dict[str, ISS.Counts] = {}
        for tf, m in masks.items():
            mask_aligned = m.reindex(sub.index, fill_value=False)
            res[tf] = ISS._count_frame(sub[mask_aligned])
        out[str(proj)] = res
    return out


@app.get("/issues-daily-dashboard")
@login_required
def issues_daily_dashboard():
    # Determine issues file from session or autodetect
    path = request.args.get("path") or _get_session_issues_path()
    if not path or not os.path.exists(path):
        # Autodetect most recent Instruction CSV in workspace using cached search
        def check_instruction(f: str) -> bool:
            fl = f.lower()
            return fl.endswith('.csv') and ('instruction' in fl or fl.startswith('csv-instruction-latest-report'))
        
        candidates = _find_files_cached(check_instruction, "instructions_csv")
        if candidates:
            path = max(candidates, key=lambda p: os.path.getmtime(p))
        else:
            path = ""
    if not path or not os.path.exists(path):
        return render_template("issues_dashboard.html", data={}, target=date.today().strftime("%d-%m-%Y")), 200
    # Target date (optional ?date= dd-mm-YYYY or ISO)
    date_q = (request.args.get("date") or "").strip()
    if date_q:
        # try our safe parser; fall back to today
        t = ISS._parse_date_safe(date_q)
        target = t or date.today()
    else:
        target = date.today()
    # Read and compute
    df = _robust_read_issues_path(path)
    # Tag External/Internal per configured name
    ext_name = (session.get("issues_external_name", DEFAULT_ISSUES_EXTERNAL) or "").strip()
    df = ISS._filter_quality(df)
    df = ISS._tag_external(df, ext_name)
    # Canonical project key - use vectorized version for better performance
    df["__ProjectKey"] = canonical_project_vectorized(df)
    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)
    masks = ISS._timeframe_masks(dates, target)
    # Build nested: { project: { 'External': {tf: Counts}, 'Internal': {tf: Counts} } }
    data: Dict[str, Dict[str, Dict[str, ISS.Counts]]] = {}
    for proj, sub in df.groupby("__ProjectKey"):
        pc: Dict[str, Dict[str, ISS.Counts]] = {}
        for cat, g in sub.groupby("__Category"):
            res: Dict[str, ISS.Counts] = {}
            for tf, m in masks.items():
                mask_aligned = m.reindex(g.index, fill_value=False)
                res[tf] = ISS._count_frame(g[mask_aligned])
            pc[str(cat)] = res
        for missing in ("External", "Internal"):
            if missing not in pc:
                pc[missing] = {tf: ISS.Counts(0, 0, 0) for tf in ("today", "month", "all")}
        data[str(proj)] = pc
    data_sorted = {k: data[k] for k in sorted(data.keys())}
    return render_template("issues_dashboard.html", data=data_sorted, target=target.strftime("%d-%m-%Y"), external_name=ext_name)


# ---------------- External Issues Report (External-only focus) ----------------

@app.get("/external-report")
@login_required
def external_report():
    """External Issues Report (site-wise), styled like the sample CSV:
    - Header with Site and date
    - Summary: Previous, Today's, Total (Total/Open/Closed/% Closed)
    - Details: grouped by 'Raised On Date' sections in descending order
    """
    # Load issues file (session or autodetect)
    path = request.args.get("path") or _get_session_issues_path()
    if not path or not os.path.exists(path):
        # Autodetect using cached search
        def check_instruction(f: str) -> bool:
            fl = f.lower()
            return fl.endswith('.csv') and ('instruction' in fl or fl.startswith('csv-instruction-latest-report'))
        
        candidates = _find_files_cached(check_instruction, "instructions_csv")
        if candidates:
            path = max(candidates, key=lambda p: os.path.getmtime(p))
        else:
            path = ""
    ext_name = (session.get("issues_external_name", DEFAULT_ISSUES_EXTERNAL) or "").strip()
    # Early render if no file
    if not path or not os.path.exists(path):
        return render_template(
            "external_report.html",
            projects=[], selected_project="", external_name=ext_name,
            header_date=date.today().strftime("%d-%m-%Y"),
            summary=None, sections=[],
        ), 200

    # Target date (for 'Today's' bucket)
    date_q = (request.args.get("date") or "").strip()
    t = ISS._parse_date_safe(date_q) if date_q else None
    target_d = t or date.today()

    df = _robust_read_issues_path(path)
    df = ISS._filter_quality(df)
    df = ISS._tag_external(df, ext_name)
    df = df[df.get("__Category") == "External"]
    df["__ProjectKey"] = canonical_project_vectorized(df)

    # Site selection
    projects = sorted([p for p in df["__ProjectKey"].astype(str).str.strip().unique() if p])
    sel = (request.args.get("project") or "").strip()
    if not sel and projects:
        sel = projects[0]
    if sel:
        df = df[df["__ProjectKey"].astype(str).str.strip().eq(sel)]

    # Parse dates safely
    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)
    masks = ISS._timeframe_masks(dates, target_d)

    # Build summary: Previous = All - Today
    all_counts = ISS._count_frame(df[masks["all"].reindex(df.index, fill_value=False)]) if not df.empty else ISS.Counts(0,0,0)
    today_counts = ISS._count_frame(df[masks["today"].reindex(df.index, fill_value=False)]) if not df.empty else ISS.Counts(0,0,0)
    prev_total = max(0, all_counts.total - today_counts.total)
    prev_open = max(0, all_counts.open - today_counts.open)
    prev_closed = max(0, all_counts.closed - today_counts.closed)
    def pct_closed(total:int, closed:int) -> float:
        return round((closed/total*100.0), 2) if total else 0.0
    summary = {
        "previous": {"total": prev_total, "open": prev_open, "closed": prev_closed, "pct": pct_closed(prev_total, prev_closed)},
        "today": {"total": today_counts.total, "open": today_counts.open, "closed": today_counts.closed, "pct": pct_closed(today_counts.total, today_counts.closed)},
        "total": {"total": all_counts.total, "open": all_counts.open, "closed": all_counts.closed, "pct": pct_closed(all_counts.total, all_counts.closed)},
    }

    # Build details grouped by date desc
    def _first_nonempty(row: pd.Series, cols: List[str]) -> str:
        for c in cols:
            if c in row and str(row[c]).strip():
                return str(row[c]).strip()
        return ""
    # Compose Location/Reference from L1..L4
    def _compose_location(row: pd.Series) -> str:
        parts = []
        for c in ("Location L1","Location L2","Location L3","Location L4"):
            v = str(row.get(c, "") or "").strip()
            if v:
                parts.append(v)
        return "/".join(parts)

    # Prepare ordering by date (newest first) and then time within date
    df = df.copy()
    df["__RaisedDate"] = dates
    # Try to keep a consistent time ordering if time column exists
    if "Raised On Time" in df.columns:
        try:
            df["__RaisedTimeOrd"] = pd.to_datetime(df["Raised On Time"], errors="coerce")
        except Exception:
            df["__RaisedTimeOrd"] = pd.NaT
    else:
        df["__RaisedTimeOrd"] = pd.NaT

    sections = []
    if not df.empty:
        groups = {}
        for idx, row in df.iterrows():
            d = row.get("__RaisedDate")
            if isinstance(d, date):
                key = d
            else:
                key = None
            groups.setdefault(key, []).append(row)
        # Sort by date desc, with None last
        ordered_keys = sorted(groups.keys(), key=lambda x: (x is None, x), reverse=True)
        for k in ordered_keys:
            rows = groups[k]
            # Sort by time desc if available, else stable
            try:
                rows_sorted = sorted(rows, key=lambda r: (r.get("__RaisedTimeOrd") if hasattr(r, "get") else None), reverse=True)
            except Exception:
                rows_sorted = rows
            sect_rows = []
            for i, r in enumerate(rows_sorted, start=1):
                sect_rows.append({
                    "sr": i,
                    "project": _first_nonempty(r, ["Project Name","Project","Location L0"]),
                    "location": _compose_location(r),
                    "description": _first_nonempty(r, ["Description","Title","Observation"]),
                    "recommendation": str(r.get("Recommendation", "") or "").strip(),
                    "raised_by": str(r.get("Raised By", "") or "").strip(),
                    "raised_on_date": (r.get("Raised On Date") or ""),
                    "raised_on_time": (r.get("Raised On Time") or ""),
                    "deadline_date": (r.get("Deadline Date") or ""),
                    "deadline_time": (r.get("Deadline Time") or ""),
                    "type": _first_nonempty(r, ["Type L0","Type","Category"]),
                    "tag1": str(r.get("Tag 1", "") or "").strip(),
                    "assigned_team": str(r.get("Assigned Team", "") or "").strip(),
                    "assigned_user": _first_nonempty(r, ["Assigned Team User","Assigned User","Assignee"]),
                    "status": _first_nonempty(r, ["Current Status","Status"]).upper(),
                })
            label = k.strftime("%d-%m-%Y") if isinstance(k, date) else ""
            sections.append({"date": label, "rows": sect_rows})

    return render_template(
        "external_report.html",
        projects=projects, selected_project=sel, external_name=ext_name,
        header_date=target_d.strftime("%d-%m-%Y"),
        summary=summary,
        sections=sections,
    )


@app.get("/external-report/export")
@login_required
def external_report_export():
    # Build a simple Excel: one sheet with External-only counts per project
    path = request.args.get("path") or _get_session_issues_path()
    if not path or not os.path.exists(path):
        return "Issues file not found", 404
    df = _robust_read_issues_path(path)
    df = ISS._filter_quality(df)
    ext_name = (session.get("issues_external_name", DEFAULT_ISSUES_EXTERNAL) or "").strip()
    df = ISS._tag_external(df, ext_name)
    df = df[df.get("__Category") == "External"]
    df["__ProjectKey"] = df.apply(canonical_project_from_row, axis=1)
    sel = (request.args.get("project") or "").strip()
    if sel:
        df = df[df["__ProjectKey"].astype(str).str.strip().eq(sel)]
    today = date.today()
    dates = df.get("Raised On Date").map(ISS._parse_date_safe) if "Raised On Date" in df.columns else pd.Series([None] * len(df), index=df.index)
    masks = ISS._timeframe_masks(dates, today)
    rec: Dict[str, str | int] = {"Project": sel or "(All)"}
    for tf, m in masks.items():
        c = ISS._count_frame(df[m.reindex(df.index, fill_value=False)])
        rec[f"{tf.title()} Total"] = c.total
        rec[f"{tf.title()} Open"] = c.open
        rec[f"{tf.title()} Closed"] = c.closed
    out_df = pd.DataFrame([rec])
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook(); ws = wb.active; ws.title = "External Issues"
    for r in dataframe_to_rows(out_df, index=False, header=True):
        ws.append(r)
    bio = io.BytesIO(); wb.save(bio); bio.seek(0)
    suffix = (sel or "All").replace("/","-")
    fname = f"External_Issues_{suffix}_{today.strftime('%Y-%m-%d')}.xlsx"
    # Save to temp file to reuse the same formatting pipeline
    tmp_path = os.path.join(tempfile.gettempdir(), f"ext_export_{uuid.uuid4().hex}.xlsx")
    with open(tmp_path, "wb") as fh:
        fh.write(bio.getvalue())
    formatted = _format_xlsx_from_path(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return send_file(formatted, as_attachment=True, download_name=fname, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------- Integrate digiqc_Inst (Combined report only) ----------------

def _inst_base_dir() -> str:
    # The repo is cloned in workspace at ./digiqc_Inst
    return os.path.join(os.getcwd(), "digiqc_Inst")


# ---------------- digiqc_Inst Combined report (Excel) ----------------

def _inst_paths() -> dict:
    base = _inst_base_dir()
    return {
        "base": base,
        "main": os.path.join(base, "main.py"),
        "combined_report": os.path.join(base, "Combined_report.xlsx"),
        "combined_summary_xlsx": os.path.join(base, "Combined_Summary.xlsx"),
    }


@app.get("/inst-report")
@login_required
def inst_report_page():
    p = _inst_paths()
    exists = {
        "main": os.path.exists(p["main"]),
        "report": os.path.exists(p["combined_report"]) or os.path.exists(p["combined_summary_xlsx"]),
    }
    return render_template("inst_report.html", exists=exists)


@app.get("/inst-report/build")
@login_required
def inst_report_build():
    p = _inst_paths()
    if not os.path.exists(p["main"]):
        return "digiqc_Inst/main.py not found", 404
    # Allow optional REPORT_DATE passthrough (?date=YYYY-MM-DD)
    env = os.environ.copy()
    report_date = (request.args.get("date") or "").strip()
    if report_date:
        env["REPORT_DATE"] = report_date
    proc = subprocess.run([os.sys.executable, p["main"]], cwd=p["base"], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return f"Combined report pipeline failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}", 500
    return redirect(url_for("inst_report_page"))


@app.get("/inst-report/download")
@login_required
def inst_report_download():
    p = _inst_paths()
    path = p["combined_report"]
    alt = p["combined_summary_xlsx"]
    today_str = date.today().strftime("%d-%m-%Y")
    dl_name = f"Activity wise stage wise instructions details {today_str}.xlsx"
    if not os.path.exists(path):
        if os.path.exists(alt):
            content = _format_xlsx_from_path(alt)
            return send_file(content, as_attachment=True, download_name=dl_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        return "Combined report not found. Run build first.", 404
    content = _format_xlsx_from_path(path)
    return send_file(content, as_attachment=True, download_name=dl_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

