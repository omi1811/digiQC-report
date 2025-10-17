#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import tempfile
from datetime import date, datetime
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import analysis_eqc as EQC
from project_utils import canonicalize_project_name, canonical_project_from_row

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
    # Projects
    df["__ProjectKey"] = df.apply(canonical_project, axis=1)
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
        month_mask = dates_sub.apply(lambda d: bool(d and d.year == target.year and d.month == target.month))
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

# Hardcoded simple credentials (no database)
AUTH_USERNAME = os.environ.get("APP_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("APP_PASSWORD", "admin123")

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            session["user"] = username
            nxt = request.args.get("next") or url_for("index")
            return redirect(nxt)
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
        return render_template("index.html")


@app.post("/daily-dashboard")
@login_required
def daily_dashboard():
    f = request.files.get("eqc")
    if not f:
        return redirect(url_for("index"))
    buf = io.BytesIO(f.read())
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
    # Avoid 405 if someone opens this URL directly; send them to the form
    return redirect(url_for("index"))


@app.post("/weekly-report")
@login_required
def weekly_report():
    f = request.files.get("eqc")
    if not f:
        return redirect(url_for("index"))
    # Use a temporary directory to avoid storing user data beyond the request.
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "Combined_EQC.csv")
        with open(in_path, "wb") as out:
            out.write(f.read())
        # Run Weekly_report in the temp dir to generate all CSVs and the combined workbook
        import subprocess, sys as _sys
        cmd = [_sys.executable, "Weekly_report.py", "--input", in_path]
        proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        if proc.returncode != 0:
            return f"Failed to build weekly workbook.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}", 500
        out_xlsx = os.path.join(os.getcwd(), "EQC_Weekly_Monthly_Cumulative_AllProjects.xlsx")
        if not os.path.exists(out_xlsx):
            return f"Workbook not found after generation.", 500
        # Send file as attachment without persisting anything else
        with open(out_xlsx, "rb") as fh:
            content = io.BytesIO(fh.read())
        # Optionally remove the generated xlsx to keep storage minimal
        try:
            os.remove(out_xlsx)
        except Exception:
            pass
        # Include current date in the download filename
        today_str = date.today().strftime("%d-%m-%Y")
        dl_name = f"EQC_Weekly_Monthly_Cumulative_AllProjects_{today_str}.xlsx"
        return send_file(content, as_attachment=True, download_name=dl_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/weekly-report")
@login_required
def weekly_report_get():
    # Avoid 405 if someone opens this URL directly; send them to the form
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
