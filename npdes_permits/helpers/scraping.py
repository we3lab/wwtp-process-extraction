"""Helper functions for processing CIWQS facility pages and downloading PDFs."""

import time
import os
import re
from urllib.parse import urlparse, parse_qs
import uuid
import tempfile
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
import pandas as pd


CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"

def abs_url(href):
    if not href or href.startswith("http"):
        return href
    if href.startswith("/"):
        return CIWQS_ROOT + href
    return f"{CIWQS_ROOT}/ciwqs/readOnly/{href}"


def hidden_fields(soup):
    return {i["name"]: i.get("value", "")
            for i in soup.find_all("input", type="hidden") if i.get("name")}


def select_value(soup, name, visible_text, *, required_label=None):
    sel = soup.find("select", {"name": name})
    if not sel:
        if required_label:
            raise RuntimeError(f"CIWQS form missing <select name={name!r}> ({required_label}).")
        return visible_text
    for opt in sel.find_all("option"):
        if opt.get_text(strip=True) == visible_text:
            return opt.get("value", visible_text)
    if required_label:
        choices = [opt.get_text(strip=True) for opt in sel.find_all("option")]
        raise RuntimeError(
            f"CIWQS {required_label}: no {visible_text!r} in <select name={name!r}>; choices={choices!r}"
        )
    return visible_text


def retry_request(session, method, url, *, data=None, max_attempts=4, timeout=120):
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.request(method, url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            print(f"[requests] {method.upper()} timed out ({attempt}/{max_attempts}): {url}")
            if attempt == max_attempts:
                raise
        except requests.exceptions.RequestException:
            raise


def new_chrome_driver(pdfs_path):
    """Chrome for facility report (after requests submits the CIWQS search)."""
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.path.abspath(pdfs_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    options.add_experimental_option("prefs", prefs)
    # "normal" waits for load event; CIWQS often omits programDrop until then (eager returns too early).
    options.page_load_strategy = "normal"
    options.add_argument("--blink-settings=imagesEnabled=false")
    user_data_dir = os.path.join(tempfile.gettempdir(), f"chrome_user_data_{uuid.uuid4().hex}")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")  # for server/SSH
    options.binary_location = "/home/daly/bin/chrome/chrome-linux64/chrome"
    service = Service("/home/daly/bin/chrome/chromedriver-linux64/chromedriver")
    return webdriver.Chrome(service=service, options=options)


# Function that selects options from filters
def selection(driver, name, text):
    select_elements = driver.find_elements(By.NAME, name)
    if not select_elements:
        return False
    select = Select(select_elements[0])
    select.select_by_visible_text(text)
    return True


def extract_reg_measure_id(href: str):
    if not href:
        return None
    try:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        if 'regMeasID' in query and query['regMeasID']:
            return str(query['regMeasID'][0])
    except Exception:
        pass
    match = re.search(r"regMeasID=(\d+)", href, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def process_all_facilities(driver, facility_url_to_facilities, output_dir, skip_keywords):
    """Main processing loop - returns failed_facilities"""
    failed_facilities = []
    reg_measure_id_to_url = {}
    url_to_pdfs = {}
    main_window = driver.window_handles[0]
    
    for idx, (facility_url, facilities) in enumerate(facility_url_to_facilities.items(), 1):
        print(f"\n[{idx}/{len(facility_url_to_facilities)}] {facilities[0]['facility_name']}")
        try:
            order_url, rm_type = find_best_order(driver, facility_url, main_window)
            if not order_url:
                print("  X No suitable active NPDES order found")
                continue
            
            # Deduplication logic
            reg_id = extract_reg_measure_id(order_url)
            if reg_id and reg_id in reg_measure_id_to_url:
                print(f"  Dedup: already processed order")
                continue
            
            if reg_id:
                reg_measure_id_to_url[reg_id] = order_url
            
            # Download PDFs
            downloaded_pdfs = download_pdfs_for_order(
                driver, order_url, facilities, output_dir, skip_keywords,
                allow_noa=(rm_type == "ENROLLEE - NPDES"), main_window=main_window
            )
            url_to_pdfs[order_url] = downloaded_pdfs
                
        except Exception as e:
            print(f"  X {e}")
            for f in facilities:
                failed_facilities.append({
                    'facility_name': f['facility_name'],
                    'agency': f['agency'],
                    'npdes_no': f['npdes_no'],
                    'region': f['region'],
                    'major_minor': f['major_minor'],
                    'order_no': f.get('order_no', ''),
                    'facility_url': facility_url,
                    'error': str(e)[:200],
                })
            try:
                if driver.current_window_handle != main_window:
                    driver.close()
                driver.switch_to.window(main_window)
            except Exception:
                pass
    
    return failed_facilities


def find_best_order(driver, facility_url, main_window):
    """Navigate to facility page, parse HTML, and return best active NPDES order.
    
    Returns: (order_url, reg_measure_type) or (None, None)
    """
    # Configuration
    TYPE_RANK = {"NPDES PERMIT": 0, "CO-PERMITTEE": 1, "ENROLLEE - NPDES": 2}
    CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
    REQUIRED_HEADERS = ['Reg Measure Type', 'Order No']
    COLUMN_CHECKS = [
        ('Status', lambda v: v.lower() == 'active'),
        ('Reg Measure Type', lambda v: v.upper() in TYPE_RANK),
    ]
    
    # Helper functions
    def abs_url(href):
        if not href or href.startswith("http"):
            return href
        if href.startswith("/"):
            return CIWQS_ROOT + href
        return f"{CIWQS_ROOT}/ciwqs/readOnly/{href}"
    
    def parse_date_safe(date_str):
        s = str(date_str).strip()
        if not s or s.lower() in ('null', 'none', 'nan', ''):
            return pd.NaT
        return pd.to_datetime(s, errors='coerce')
    
    # Navigate to facility page
    driver.execute_script(f"window.open('{facility_url}', '_blank');")
    time.sleep(1)
    new_win = [h for h in driver.window_handles if h != main_window][0]
    driver.switch_to.window(new_win)
    
    WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
    time.sleep(1)
    
    page_html = driver.page_source
    driver.close()
    driver.switch_to.window(main_window)
    
    # Parse HTML for best order
    soup = BeautifulSoup(page_html, "html.parser")
    for table in soup.find_all('table'):
        all_rows = table.find_all('tr')
        
        # Find header row
        for hdr_idx, row in enumerate(all_rows[:4]):
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]
            
            # Validate header row
            if len(texts) < 5 or any(len(t) > 80 for t in texts):
                continue
            if not all(any(req in t for t in texts) for req in REQUIRED_HEADERS):
                continue
            
            col_index = {t: i for i, t in enumerate(texts)}
            best_href, best_type_rank = None, 99
            best_effective, best_rm_type = pd.NaT, None
            
            # Process data rows
            for data_row in all_rows[hdr_idx + 1:]:
                dcells = data_row.find_all('td')
                if not dcells:
                    continue
                
                def gc(col_name):
                    i = col_index.get(col_name, -1)
                    return dcells[i].get_text(strip=True) if 0 <= i < len(dcells) else ""
                
                # Apply validation checks
                if not all(check_fn(gc(col_name)) for col_name, check_fn in COLUMN_CHECKS):
                    continue
                
                # Extract order URL
                order_idx = col_index.get('Order No.', -1)
                if order_idx < 0 or order_idx >= len(dcells):
                    continue
                
                a_tag = dcells[order_idx].find('a', href=True)
                if not a_tag:
                    continue
                
                href = abs_url(a_tag['href'])
                if not href:
                    continue
                
                # Calculate priority
                rm_type = gc('Reg Measure Type').upper()
                type_rank = TYPE_RANK[rm_type]
                effective_dt = parse_date_safe(gc('Effective Date'))
                
                if pd.isna(effective_dt):
                    continue
                
                # Update best if higher priority
                if (type_rank < best_type_rank or 
                    (type_rank == best_type_rank and effective_dt > best_effective)):
                    best_href = href
                    best_type_rank = type_rank
                    best_effective = effective_dt
                    best_rm_type = rm_type
            
            if best_href:
                eff_str = best_effective.date() if not pd.isna(best_effective) else 'N/A'
                print(f"  Best order: {best_rm_type}, rank={best_type_rank}, effective={eff_str}")
                return best_href, best_rm_type
    
    return None, None


def download_pdfs_for_order(driver, order_url, facilities, output_dir, skip_keywords, allow_noa=False, main_window=None):
    """Download PDFs from an order page, applying keyword filters.
    
    Returns: list of downloaded PDF filenames
    """
    if main_window is None:
        main_window = driver.window_handles[0]
    
    downloaded_pdfs = []
    
    # Adjust skip keywords based on allow_noa
    if allow_noa:
        noa_set = {'noa', 'noi'}
        active_skip = {
            'embedded': [k for k in skip_keywords['embedded'] 
                        if not any(n in k.lower() for n in noa_set)],
            'beginning': [p for p in skip_keywords['beginning'] 
                         if not any(p.lower().startswith(n) for n in noa_set)]
        }
    else:
        active_skip = skip_keywords
    
    print(f"\n  Opening order URL...")
    driver.execute_script(f"window.open('{order_url}', '_blank');")
    time.sleep(1)
    
    new_window = [h for h in driver.window_handles if h != main_window][0]
    driver.switch_to.window(new_window)
    time.sleep(3)
    
    # Find all PDF links (including "Information Sheet" links)
    pdf_documents = driver.find_elements(
        By.XPATH, 
        "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
    )
    
    # Add "Information Sheet" PDF links
    info_sheet_docs = driver.find_elements(
        By.XPATH,
        "//a[contains(translate(@href,'PDF','pdf'),'.pdf') and "
        "contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'information sheet')]"
    )
    
    seen_hrefs = {d.get_attribute('href') for d in pdf_documents}
    pdf_documents.extend([d for d in info_sheet_docs if d.get_attribute('href') not in seen_hrefs])
    
    if len(pdf_documents) == 0:
        pdf_documents = driver.find_elements(
            By.XPATH, 
            "//a[contains(translate(@href, 'PDF', 'pdf'), '.pdf')]"
        )
    
    print(f"  Found {len(pdf_documents)} PDFs on page")
    
    for pdf_element in pdf_documents:
        try:
            pdf_name = pdf_element.text
            
            # Check if PDF should be skipped
            pdf_lower = pdf_name.lower()
            if any(keyword.lower() in pdf_lower for keyword in active_skip['embedded']):
                print(f"        Skipping: {pdf_name}")
                continue
            if any(pdf_lower.startswith(pattern.lower()) for pattern in active_skip['beginning']):
                print(f"        Skipping: {pdf_name}")
                continue
            
            print(f"        Downloading: {pdf_name}")
            
            # Click and wait for download
            before = set([f for f in os.listdir(output_dir) if f.lower().endswith('.pdf')])
            pdf_element.click()
            
            # Wait for new PDF to appear and stabilize
            end = time.time() + 60
            new_file = None
            while time.time() < end:
                current = set([f for f in os.listdir(output_dir) if f.lower().endswith('.pdf')])
                new_files = [f for f in current - before 
                            if not f.lower().endswith('.crdownload')]
                if new_files:
                    newest = max(new_files, 
                                key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
                    path = os.path.join(output_dir, newest)
                    try:
                        s1 = os.path.getsize(path)
                        time.sleep(0.5)
                        s2 = os.path.getsize(path)
                        if s1 == s2 and s1 > 0:
                            new_file = newest
                            break
                    except OSError:
                        pass
                time.sleep(0.5)
            
            if not new_file:
                print("        X Download did not complete or file not detected")
                continue
            
            downloaded_pdfs.append(new_file)
                
        except Exception as e:
            print(f"        X Download failed: {e}")
    
    driver.close()
    driver.switch_to.window(main_window)
    
    return downloaded_pdfs


def extract_reg_measure_id(href):
    """Extract regMeasID from URL query string or path."""
    if not href:
        return None
    try:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        if 'regMeasID' in query and query['regMeasID']:
            return str(query['regMeasID'][0])
    except Exception:
        pass
    match = re.search(r"regMeasID=(\d+)", href, re.IGNORECASE)
    return match.group(1) if match else None