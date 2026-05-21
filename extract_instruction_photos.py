#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

try:
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:  # pragma: no cover - optional dependency
    ChromeDriverManager = None


IMAGE_EXT_RE = r"(?:jpg|jpeg|png|webp|gif|bmp|heic|heif)"
URL_RE = re.compile(rf"https?://[^\s\"'<>]+(?:\.{IMAGE_EXT_RE})(?:\?[^\s\"'<>]*)?", re.IGNORECASE)
GENERIC_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

SECTION_TO_COLUMN = {
    "issue details": "Raised Photo",
    "raised details": "Raised Photo",
    "response details": "Responded Photo",
    "responded details": "Responded Photo",
    "closed details": "Closure Photo",
    "closure details": "Closure Photo",
}

RAISED_HINTS = [
    "raised photo",
    "observation photo",
    "issue photo",
    "before photo",
    "raised_images",
    "raisedimages",
]
RESP_CLOSED_HINTS = [
    "closed photo",
    "responded photo",
    "response photo",
    "after photo",
    "closure photo",
    "resolved photo",
    "rectification photo",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract Raised, Responded, and Closure photo URLs from Instructions CSV. "
            "If photo columns are not present, optional URL scraping can be used."
        )
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to Instructions CSV (for example CSV-INSTRUCTION-LATEST-REPORT....csv)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="instructions_photo_links.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--fetch-from-url",
        action="store_true",
        help="Fetch each issue URL and try to infer Raised/Responded/Closure photo links from page HTML.",
    )
    p.add_argument(
        "--use-browser-render",
        action="store_true",
        help="When fetching URLs, also try a Selenium-rendered browser page source if requests alone is insufficient.",
    )
    p.add_argument(
        "--browser-profile-dir",
        default="",
        help="Optional Chrome user-data-dir to reuse an authenticated browser profile for Selenium rendering.",
    )
    p.add_argument(
        "--interactive-login",
        action="store_true",
        help="Open Chrome with a disposable profile, let the user log in manually, and reuse that session for extraction.",
    )
    p.add_argument(
        "--login-url",
        default="https://app.digiqc.com/",
        help="URL to open for the manual digiQC login flow.",
    )
    p.add_argument(
        "--cookie",
        default="",
        help="Optional raw Cookie header value for authenticated issue URL access.",
    )
    p.add_argument(
        "--bearer-token",
        default="",
        help="Optional bearer token for authenticated requests.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout seconds when --fetch-from-url is used.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to process (0 means all).",
    )
    return p.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for sep in (",", "\t", None):
        try:
            if sep is None:
                return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
            return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, engine="python")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not read CSV: {path}. Last error: {last_err}")


def split_links(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    candidates = GENERIC_URL_RE.findall(text)
    if candidates:
        return list(dict.fromkeys(candidates))
    parts = re.split(r"[\n;,|]+", text)
    out = [p.strip() for p in parts if p.strip().lower().startswith("http")]
    return list(dict.fromkeys(out))


def detect_photo_columns(columns: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    raised_cols: List[str] = []
    closed_cols: List[str] = []
    generic_cols: List[str] = []
    for col in columns:
        norm = col.strip().lower()
        if "photo" not in norm and "image" not in norm and "attachment" not in norm:
            continue
        if any(k in norm for k in ("raised", "before", "observation", "issue")):
            raised_cols.append(col)
        elif any(k in norm for k in ("closed", "responded", "response", "after", "resolve", "rectif", "closure")):
            closed_cols.append(col)
        else:
            generic_cols.append(col)
    return raised_cols, closed_cols, generic_cols


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def is_media_candidate(tag: str, url: str) -> bool:
    if not url:
        return False
    tag = tag.lower()
    url_l = url.lower()
    if tag in {"img", "source", "video", "picture"}:
        return True
    if re.search(rf"\.(?:{IMAGE_EXT_RE})(?:\?|$)", url_l, re.IGNORECASE):
        return True
    return any(token in url_l for token in ("image", "photo", "attachment", "media", "blob", "upload"))


def build_chrome_options(headless: bool, browser_profile_dir: str = "") -> Options:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    if browser_profile_dir.strip():
        options.add_argument(f"--user-data-dir={browser_profile_dir.strip()}")
    return options


def create_webdriver(headless: bool, browser_profile_dir: str = ""):
    if ChromeDriverManager is None:
        raise RuntimeError("webdriver-manager is not installed; browser rendering fallback is unavailable")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=build_chrome_options(headless, browser_profile_dir))


def wait_for_ready(driver, timeout: int) -> None:
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")


def login_interactively(driver, login_url: str, timeout: int) -> None:
    driver.get(login_url)
    wait_for_ready(driver, timeout)
    print("Open the browser window, sign in to digiQC, then return here and press Enter to continue.")
    input()


class SectionPhotoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_section = ""
        self.photos: Dict[str, List[str]] = {
            "Raised Photo": [],
            "Responded Photo": [],
            "Closure Photo": [],
            "Unclassified": [],
        }

    def handle_data(self, data: str) -> None:
        text = normalize_text(data)
        if not text:
            return
        for marker, column_name in SECTION_TO_COLUMN.items():
            if marker in text:
                self.current_section = column_name
                break

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = {key.lower(): value for key, value in attrs}
        candidate = ""
        for key in ("src", "href", "data-src", "data-original", "data-lazy-src", "data-full", "data-url"):
            raw_value = attrs_dict.get(key)
            if raw_value:
                matches = GENERIC_URL_RE.findall(raw_value)
                if matches:
                    candidate = matches[0]
                    break
        if not candidate and attrs_dict.get("srcset"):
            matches = GENERIC_URL_RE.findall(attrs_dict["srcset"])
            if matches:
                candidate = matches[0]
        if candidate and is_media_candidate(tag, candidate):
            section = self.current_section or "Unclassified"
            self.photos.setdefault(section, []).append(candidate)


def extract_sectioned_photo_urls(html: str) -> Dict[str, List[str]]:
    parser = SectionPhotoParser()
    parser.feed(html or "")
    parser.close()
    return {key: list(dict.fromkeys(values)) for key, values in parser.photos.items()}


def classify_urls_from_html(html: str) -> Tuple[List[str], List[str], List[str]]:
    sectioned = extract_sectioned_photo_urls(html)
    raised = sectioned.get("Raised Photo", [])
    closed = sectioned.get("Responded Photo", []) + sectioned.get("Closure Photo", [])
    if raised or closed:
        return raised, closed, sectioned.get("Unclassified", [])

    text = html or ""
    lower = text.lower()
    all_urls = sorted(set(URL_RE.findall(text)))
    if not all_urls:
        all_urls = sorted(set(u for u in GENERIC_URL_RE.findall(text) if re.search(rf"\.{IMAGE_EXT_RE}(?:\?|$)", u, re.IGNORECASE)))

    parsed_raised: List[str] = []
    parsed_closed: List[str] = []
    unknown: List[str] = []
    for url in all_urls:
        idx = lower.find(url.lower())
        if idx == -1:
            unknown.append(url)
            continue
        start = max(0, idx - 220)
        end = min(len(lower), idx + len(url) + 220)
        window = lower[start:end]
        if any(k in window for k in RAISED_HINTS):
            parsed_raised.append(url)
        elif any(k in window for k in RESP_CLOSED_HINTS):
            parsed_closed.append(url)
        else:
            unknown.append(url)
    return list(dict.fromkeys(parsed_raised)), list(dict.fromkeys(parsed_closed)), list(dict.fromkeys(unknown))


def fetch_html_with_browser(
    url: str,
    timeout: int,
    browser_profile_dir: str = "",
    driver=None,
) -> str:
    own_driver = driver is None
    if own_driver:
        driver = create_webdriver(headless=True, browser_profile_dir=browser_profile_dir)
    try:
        driver.get(url)
        wait_for_ready(driver, timeout)
        return driver.page_source
    finally:
        if own_driver and driver is not None:
            driver.quit()


def fetch_and_extract(
    url: str,
    session: requests.Session,
    timeout: int,
    use_browser: bool = False,
    browser_profile_dir: str = "",
    browser_driver=None,
) -> Tuple[List[str], List[str], str]:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code >= 400:
            return [], [], f"HTTP {r.status_code}"
        raised, closed, unknown = classify_urls_from_html(r.text)
        if raised or closed:
            return raised + unknown, closed, "OK"
        if use_browser:
            browser_html = fetch_html_with_browser(url, timeout, browser_profile_dir, driver=browser_driver)
            raised_b, closed_b, unknown_b = classify_urls_from_html(browser_html)
            if raised_b or closed_b or unknown_b:
                return raised_b + unknown_b, closed_b, "OK (browser render)"
        if unknown:
            return unknown, [], "Unclassified image links found"
        return [], [], "No photo links found"
    except Exception as exc:
        return [], [], str(exc)


def to_pipe(values: Sequence[str]) -> str:
    vals = [v.strip() for v in values if str(v).strip()]
    return " | ".join(list(dict.fromkeys(vals)))


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = read_csv_robust(in_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    raised_cols, closed_cols, generic_cols = detect_photo_columns(df.columns)

    url_col = "URL" if "URL" in df.columns else "Url" if "Url" in df.columns else "url" if "url" in df.columns else ""
    status_col = "Current Status" if "Current Status" in df.columns else ""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }
    )
    if args.cookie.strip():
        session.headers.update({"Cookie": args.cookie.strip()})
    if args.bearer_token.strip():
        session.headers.update({"Authorization": f"Bearer {args.bearer_token.strip()}"})

    browser_driver = None
    temp_profile_dir: Optional[str] = None
    effective_browser_profile_dir = args.browser_profile_dir.strip()
    try:
        if args.interactive_login:
            if not effective_browser_profile_dir:
                temp_profile_dir = tempfile.mkdtemp(prefix="digiqc-login-")
                effective_browser_profile_dir = temp_profile_dir
            browser_driver = create_webdriver(headless=False, browser_profile_dir=effective_browser_profile_dir)
            login_interactively(browser_driver, args.login_url, args.timeout)
        elif args.use_browser_render:
            browser_driver = create_webdriver(headless=True, browser_profile_dir=effective_browser_profile_dir)

        rows: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            ref = str(row.get("Reference ID", "")).strip()
            status = str(row.get(status_col, "")).strip().upper() if status_col else ""
            issue_url = str(row.get(url_col, "")).strip() if url_col else ""

            raised_links: List[str] = []
            responded_links: List[str] = []
            closure_links: List[str] = []
            notes: List[str] = []

            for c in raised_cols:
                raised_links.extend(split_links(str(row.get(c, ""))))
            for c in closed_cols:
                linked = split_links(str(row.get(c, "")))
                if status == "RESPONDED":
                    responded_links.extend(linked)
                else:
                    closure_links.extend(linked)

            # Use generic media columns as fallback by status.
            if generic_cols:
                generic_links: List[str] = []
                for c in generic_cols:
                    generic_links.extend(split_links(str(row.get(c, ""))))
                if generic_links:
                    if status == "RESPONDED":
                        responded_links.extend(generic_links)
                    elif status == "CLOSED":
                        closure_links.extend(generic_links)
                    else:
                        raised_links.extend(generic_links)

            if args.fetch_from_url and issue_url:
                fetch_raised, fetch_closed, msg = fetch_and_extract(
                    issue_url,
                    session,
                    args.timeout,
                    use_browser=args.use_browser_render or args.interactive_login,
                    browser_profile_dir=effective_browser_profile_dir,
                    browser_driver=browser_driver,
                )
                if fetch_raised:
                    raised_links.extend(fetch_raised)
                if fetch_closed:
                    if status == "RESPONDED":
                        responded_links.extend(fetch_closed)
                    elif status == "CLOSED":
                        closure_links.extend(fetch_closed)
                    else:
                        closure_links.extend(fetch_closed)
                notes.append(msg)

            if not raised_links and not responded_links and not closure_links:
                if raised_cols or closed_cols or generic_cols:
                    notes.append("No photo links found in detected photo columns")
                elif not args.fetch_from_url:
                    notes.append("No photo columns in CSV (use --fetch-from-url to scrape issue links)")
                else:
                    notes.append("No photo links extracted")

            rows.append(
                {
                    "Reference ID": ref,
                    "Current Status": status,
                    "Issue URL": issue_url,
                    "Raised Photo": to_pipe(raised_links),
                    "Responded Photo": to_pipe(responded_links),
                    "Closure Photo": to_pipe(closure_links),
                    "Notes": " ; ".join(n for n in notes if n),
                }
            )

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

        print(f"Input rows processed: {len(df)}")
        print(f"Detected raised-photo columns: {raised_cols or 'None'}")
        print(f"Detected closed/responded-photo columns: {closed_cols or 'None'}")
        print(f"Detected generic media columns: {generic_cols or 'None'}")
        print(f"Browser render fallback: {'enabled' if args.use_browser_render or args.interactive_login else 'disabled'}")
        print(f"Output written: {out_path.resolve()}")
    finally:
        if browser_driver is not None:
            browser_driver.quit()
        if temp_profile_dir:
            shutil.rmtree(temp_profile_dir, ignore_errors=True)


if __name__ == "__main__":
    main()