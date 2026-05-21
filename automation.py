import os
import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib, ssl
from dotenv import load_dotenv
import schedule
import subprocess
import sys
from pathlib import Path

# Load environment variables
load_dotenv()

class DigiQCAutomator:
    def __init__(self):
        self.driver = None
        self.start_date = os.getenv('REPORT_START_DATE', '01/02/2025')
        self.download_dir = Path(os.getenv('DOWNLOAD_FOLDER', './downloads')).resolve()
        self.dashboard_dir = Path(os.getenv('DASHBOARD_DIR', '.')).resolve()
        self.combined_dir = self.dashboard_dir  # Where Combined_*.csv should live
        os.makedirs(self.download_dir, exist_ok=True)
        
    def setup_driver(self):
        """Initialize Chrome WebDriver with download preferences"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True
        })
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.driver.implicitly_wait(10)
        
    def login(self):
        """Login to digiQC web app"""
        self.driver.get("https://app.digiqc.com/")
        wait = WebDriverWait(self.driver, 30)
        
        # Flexible selector strategy for email field
        email_selectors = [
            "//input[@type='email']",
            "//input[@name='email']",
            "//input[contains(@placeholder, 'email') or contains(@placeholder, 'Email')]",
            "//input[@id='email']"
        ]
        for selector in email_selectors:
            try:
                email_field = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                email_field.send_keys(os.getenv('DIGIQC_EMAIL'))
                break
            except:
                continue
        
        # Handle multi-step login if present
        try:
            continue_btn = self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Continue') or contains(text(), 'Next') or @type='button']")
            if continue_btn.is_displayed():
                continue_btn.click()
                time.sleep(2)
        except:
            pass
            
        # Password field
        password_selectors = [
            "//input[@type='password']",
            "//input[@name='password']",
            "//input[contains(@placeholder, 'password') or contains(@placeholder, 'Password')]"
        ]
        for selector in password_selectors:
            try:
                password_field = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                password_field.send_keys(os.getenv('DIGIQC_PASSWORD'))
                break
            except:
                continue
        
        # Login button
        login_selectors = [
            "//button[@type='submit']",
            "//button[contains(text(), 'Login') or contains(text(), 'Sign In') or contains(text(), 'Log In')]",
            "//input[@type='submit']"
        ]
        for selector in login_selectors:
            try:
                login_btn = self.driver.find_element(By.XPATH, selector)
                login_btn.click()
                break
            except:
                continue
        
        time.sleep(5)  # Wait for redirect
        
    def navigate_to_reports(self):
        """Navigate to Reports section"""
        wait = WebDriverWait(self.driver, 20)
        
        # Try to open sidebar if collapsed
        try:
            menu_btn = wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR, 
                "button[aria-label='Menu'], .menu-toggle, .hamburger, [data-testid='menu-button']")))
            menu_btn.click()
            time.sleep(1)
        except:
            pass
            
        # Click Reports link
        reports_selectors = [
            "//a[contains(text(), 'Reports')]",
            "//span[contains(text(), 'Reports')]/parent::a",
            "//div[contains(text(), 'Reports') and @role='menuitem']",
            "//li[contains(text(), 'Reports')]"
        ]
        for selector in reports_selectors:
            try:
                reports_link = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                reports_link.click()
                time.sleep(3)
                return
            except:
                continue
        raise Exception("Could not find Reports navigation link")
        
    def _set_date_filter(self, start_date: str, end_date: str):
        """Helper to set date range filters"""
        wait = WebDriverWait(self.driver, 15)
        
        # Try common date input patterns
        date_selectors = [
            ("//input[@id='startDate' or @name='start_date' or @placeholder*='Start' or @data-testid='start-date']", start_date),
            ("//input[@id='endDate' or @name='end_date' or @placeholder*='End' or @data-testid='end-date']", end_date),
            ("//input[contains(@class, 'start-date')]", start_date),
            ("//input[contains(@class, 'end-date')]", end_date),
        ]
        
        for selector, value in date_selectors:
            try:
                field = self.driver.find_element(By.XPATH, selector)
                field.clear()
                field.send_keys(value)
            except:
                continue
                
    def _click_download(self):
        """Click download/export button"""
        wait = WebDriverWait(self.driver, 15)
        download_selectors = [
            "//button[contains(text(), 'Download') or contains(text(), 'Export') or contains(@title, 'Download')]",
            "//a[@download]",
            "//button[contains(@class, 'download') or contains(@class, 'export')]",
            "//i[contains(@class, 'download') or contains(@class, 'export')]/parent::button"
        ]
        for selector in download_selectors:
            try:
                btn = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                btn.click()
                return True
            except:
                continue
        return False
        
    def download_report(self, report_name: str, start_date: str, end_date: str, output_filename: str):
        """Download a specific report and save with known filename"""
        wait = WebDriverWait(self.driver, 20)
        
        # Find and click the report type
        report_selectors = [
            f"//span[contains(text(), '{report_name}')]",
            f"//div[contains(text(), '{report_name}')]",
            f"//a[contains(text(), '{report_name}')]",
            f"//label[contains(text(), '{report_name}')]/preceding-sibling::input"
        ]
        
        report_found = False
        for selector in report_selectors:
            try:
                report_elem = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                # If it's a checkbox/radio, select it; otherwise click to navigate
                if report_elem.tag_name in ['input']:
                    if not report_elem.is_selected():
                        self.driver.execute_script("arguments[0].click();", report_elem)
                else:
                    report_elem.click()
                report_found = True
                break
            except:
                continue
        
        if not report_found:
            print(f"⚠️ Could not find report '{report_name}' - skipping")
            return False
            
        time.sleep(2)
        
        # Set date filters
        self._set_date_filter(start_date, end_date)
        time.sleep(1)
        
        # Click download
        if not self._click_download():
            print(f"⚠️ Could not trigger download for '{report_name}'")
            return False
            
        # Wait for file to appear
        time.sleep(8)  # Adjust based on typical download time
        
        # Find the downloaded file and rename to expected name
        downloaded = self._get_latest_download(report_name.lower())
        if downloaded:
            target_path = self.download_dir / output_filename
            downloaded.rename(target_path)
            print(f"✅ Saved: {output_filename}")
            return True
        else:
            print(f"⚠️ Download file not found for '{report_name}'")
            return False
            
    def _get_latest_download(self, keyword: str) -> Path | None:
        """Find most recent file matching keyword in download dir"""
        files = [f for f in self.download_dir.iterdir() 
                if f.suffix.lower() == '.csv' and keyword in f.name.lower()]
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def prepare_combined_files(self):
        """Ensure Combined_EQC.csv and Combined_Instructions.csv exist for dashboard"""
        # The automation downloads directly to Combined_*.csv names
        # If your workflow requires merging multiple files, add logic here:
        # Example: pd.concat([pd.read_csv(f) for f in raw_files]).to_csv(combined_path)
        
        eqc_combined = self.download_dir / "Combined_EQC.csv"
        instr_combined = self.download_dir / "Combined_Instructions.csv"
        
        if not eqc_combined.exists():
            print(f"❌ Missing: {eqc_combined}")
            return False
        if not instr_combined.exists():
            print(f"❌ Missing: {instr_combined}")
            return False
            
        # Copy to dashboard directory if different
        if self.combined_dir != self.download_dir:
            import shutil
            shutil.copy2(eqc_combined, self.combined_dir / "Combined_EQC.csv")
            shutil.copy2(instr_combined, self.combined_dir / "Combined_Instructions.csv")
            
        return True
    
    def run_dashboard_script(self) -> str:
        """Execute your build_dashboard.py and capture output"""
        dashboard_script = self.dashboard_dir / "build_dashboard.py"
        if not dashboard_script.exists():
            raise FileNotFoundError(f"Dashboard script not found: {dashboard_script}")
        
        # Run your script and capture stdout
        result = subprocess.run(
            [sys.executable, str(dashboard_script), 
             "--eqc", str(self.combined_dir / "Combined_EQC.csv"),
             "--date", datetime.now().strftime("%d-%m-%Y")],
            cwd=str(self.dashboard_dir),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Dashboard script error:\n{result.stderr}")
            raise RuntimeError(f"Dashboard generation failed: {result.stderr}")
            
        return result.stdout  # This is your console dashboard output
    
    def console_to_html(self, console_output: str) -> str:
        """Convert plain text console output to styled HTML email"""
        # Escape HTML special chars
        escaped = console_output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Convert to pre-formatted HTML with styling
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>digiQC Daily Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 20px; text-align: center; }}
                .header h1 {{ margin: 0 0 5px 0; font-size: 24px; }}
                .header p {{ margin: 0; opacity: 0.9; }}
                .content {{ padding: 25px; }}
                pre {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 15px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.5; }}
                .footer {{ background: #f8f9fa; padding: 15px 25px; text-align: center; color: #6c757d; font-size: 12px; border-top: 1px solid #e9ecef; }}
                .metric {{ color: #007bff; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏗️ digiQC Daily Quality Dashboard</h1>
                    <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Period: {os.getenv('REPORT_START_DATE')} to {datetime.now().strftime('%m/%d/%Y')}</p>
                </div>
                <div class="content">
                    <pre>{escaped}</pre>
                </div>
                <div class="footer">
                    Automated report from digiQC • Questions? Contact your QA team
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def send_email(self, subject: str, html_content: str, attachments: list = None):
        """Send email with HTML body and optional attachments"""
        sender = os.getenv('EMAIL_SENDER')
        recipients = [r.strip() for r in os.getenv('EMAIL_RECIPIENTS', '').split(',') if r.strip()]
        password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 465))
        
        if not recipients:
            print("⚠️ No email recipients configured")
            return False
        
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = ", ".join(recipients)
        
        # Attach HTML content
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)
        
        # Attach CSV files if provided
        if attachments:
            for filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        filename = os.path.basename(filepath)
                        part.add_header("Content-Disposition", f"attachment; filename= {filename}")
                        message.attach(part)
        
        # Send via SMTP
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(sender, password)
                server.sendmail(sender, recipients, message.as_string())
            print("✅ Email sent successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
    
    def run_daily_task(self):
        """Execute the full automation workflow"""
        print(f"🔄 Starting digiQC automation at {datetime.now()}")
        
        try:
            # 1. Setup browser
            self.setup_driver()
            
            # 2. Login and navigate
            self.login()
            self.navigate_to_reports()
            
            # 3. Calculate date range
            end_date = datetime.now().strftime('%m/%d/%Y')
            start_date = self.start_date  # From .env: 01/02/2025
            
            # 4. Download reports with expected filenames
            print("📥 Downloading EQC report...")
            self.download_report("EQC Latest log", start_date, end_date, "Combined_EQC.csv")
            
            print("📥 Downloading Instructions report...")
            self.download_report("Instructions Latest log", start_date, end_date, "Combined_Instructions.csv")
            
            # 5. Verify files exist for dashboard
            if not self.prepare_combined_files():
                raise FileNotFoundError("Required CSV files not found for dashboard generation")
            
            # 6. Generate dashboard using YOUR existing code
            print("📊 Generating dashboard...")
            dashboard_output = self.run_dashboard_script()
            
            # 7. Convert to HTML email
            html_email = self.console_to_html(dashboard_output)
            
            # 8. Send email with dashboard + CSV attachments
            subject = f"🏗️ digiQC Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
            attachments = [
                str(self.combined_dir / "Combined_EQC.csv"),
                str(self.combined_dir / "Combined_Instructions.csv")
            ]
            self.send_email(subject, html_email, attachments)
            
            print("✅ Daily task completed successfully!")
            return True
            
        except Exception as e:
            error_msg = f"❌ Automation failed: {type(e).__name__}: {e}"
            print(error_msg)
            # Optional: Send error notification
            try:
                error_html = f"<pre>{error_msg}\n\nTraceback available in logs.</pre>"
                self.send_email(
                    f"⚠️ digiQC Automation Error - {datetime.now().strftime('%Y-%m-%d')}",
                    error_html
                )
            except:
                pass
            return False
        finally:
            if self.driver:
                self.driver.quit()
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()


def job():
    """Wrapper for scheduler"""
    automator = DigiQCAutomator()
    success = automator.run_daily_task()
    automator.close()
    return success


if __name__ == "__main__":
    import sys
    
    # Allow manual run with optional date override
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("🚀 Running digiQC automation (manual mode)...")
        job()
    else:
        # Schedule for daily execution
        run_time = os.getenv('SCHEDULE_TIME', '08:00')
        print(f"⏰ Scheduled for daily execution at {run_time}")
        schedule.every().day.at(run_time).do(job)
        
        # Keep script running
        while True:
            schedule.run_pending()
            time.sleep(60)