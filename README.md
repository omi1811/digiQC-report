# digiQC-report

Minimal Flask app + reporting utilities to build EQC dashboards and weekly/monthly/cumulative Excel reports from digiQC exports. Includes simple login.

## Deploy to Render

We’ve included a `render.yaml` for one-click deploy.

1) Push this repo to GitHub (already configured)
2) Create a new Web Service on Render and pick this repo
3) Render will auto-detect `render.yaml`
4) Set the following environment variables in Render (Environment tab):
	 - SECRET_KEY: a long random string
	 - APP_USERNAME: your login username
	 - APP_PASSWORD: your login password

It runs with:

	- Build: `pip install -r requirements.txt`
	- Start: `gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT app:app`

Notes:
- Free plan sleeps on inactivity; first request after sleep can take ~30s to wake.
- Max request duration is limited; large Excel generations are done in-memory and streamed back.

## Local development

Install dependencies and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY=dev
export APP_USERNAME=admin
export APP_PASSWORD=admin123
python app.py
```

Visit http://localhost:5000, sign in, and upload your Combined_EQC export.

## Environment variables

- SECRET_KEY: Flask session secret (required in prod)
- APP_USERNAME / APP_PASSWORD: simple auth credentials

## Project layout

- `app.py`: Flask app with two flows: Daily Dashboard (web) and Weekly Report (download)
- `analysis_eqc.py`: core logic for stage normalization and counts
- `Weekly_report.py`: generates weekly/monthly/cumulative CSVs and triggers combiner
- `combine_reports_to_excel.py`: combines generated CSVs to a formatted workbook
- `templates/`: Bootstrap-based UI
- `archive/`: older and auxiliary scripts