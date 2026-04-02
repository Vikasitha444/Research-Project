"""
TopJobs.lk - Job Vacancy Scraper
Category: Software Development & QA (FA=SDQ)
------------------------------------------
Usage:
    python topjobs_scraper.py                    # GUI folder picker open vena
    python topjobs_scraper.py --fa SDQ           # category change karana
    python topjobs_scraper.py --fa ITM           # IT Management
    python topjobs_scraper.py --out C:/MyFolder  # folder manually dena (GUI skip)

CSV auto save karana — topjobs_<FA>_<timestamp>.csv
"""

import os
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import csv
import argparse
import sys
from datetime import datetime

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
BASE_URL = "https://www.topjobs.lk"
LIST_URL = f"{BASE_URL}/applicant/vacancybyfunctionalarea.jsp"
JOB_URL  = f"{BASE_URL}/employer/JobAdvertismentServlet"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         BASE_URL + "/",
}

# Exact CSV column order
CSV_COLUMNS = [
    "job_number",
    "title",
    "company",
    "job_code",
    "employer_code",
    "ad_code",
    "description",
    "opening_date",
    "closing_date",
    "location",
    "url",
    "scraped_at",
    "functional_area",
]

# ──────────────────────────────────────────────
#  FOLDER PICKER  (tkinter GUI)
# ──────────────────────────────────────────────
def pick_save_folder() -> str:
    """
    GUI folder picker dialog ekak open karana.
    Cancel karata, current working directory use karana.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                    # main window hide karana
        root.attributes("-topmost", True)  # dialog screen front ekate ganna

        folder = filedialog.askdirectory(
            title="CSV Save Karana Folder Ekak Select Karanna",
            initialdir=os.path.expanduser("~"),  # home folder dari open karana
        )
        root.destroy()

        if folder:
            print(f"📁  Save location: {folder}")
            return folder
        else:
            cwd = os.getcwd()
            print(f"⚠  Folder එකක් Select කළේ නැති නිසා, මේ Folder එකටම Save කරන්නම්: {cwd}")
            return cwd

    except Exception as e:
        cwd = os.getcwd()
        print(f"⚠  Folder picker open wunane na ({e})")
        print(f"   Current folder use karana: {cwd}")
        return cwd


# ──────────────────────────────────────────────
#  SCRAPER
# ──────────────────────────────────────────────
def fetch_page(fa: str) -> BeautifulSoup:
    """TopJobs page HTML gaana."""
    try:
        resp = requests.get(LIST_URL, params={"FA": fa}, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Page load failed: {e}")
        sys.exit(1)
    return BeautifulSoup(resp.text, "html.parser")


def parse_jobs(soup: BeautifulSoup, fa: str) -> list[dict]:
    """
    <tr id="tr0">, <tr id="tr1"> ... rows parse karana.
    CSV_COLUMNS widin exact field names use karana.
    """
    jobs       = []
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = soup.find_all(
        "tr",
        id=lambda x: x and x.startswith("tr") and x[2:].isdigit()
    )

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        # ── job_number  (column 1)
        job_number = cells[0].get_text(strip=True)

        # ── info cell  (column 3 — title, company, hidden codes)
        info_cell = cells[2]

        title_tag = info_cell.find("h2")
        comp_tag  = info_cell.find("h1")
        title     = title_tag.get_text(strip=True) if title_tag else ""
        company   = comp_tag.get_text(strip=True)  if comp_tag  else ""

        # hidden spans: hdnJC=job_code, hdnEC=employer_code, hdnAC=ad_code
        jc_span = info_cell.find("span", id=lambda x: x and x.startswith("hdnJC"))
        ec_span = info_cell.find("span", id=lambda x: x and x.startswith("hdnEC"))
        ac_span = info_cell.find("span", id=lambda x: x and x.startswith("hdnAC"))

        job_code      = jc_span.get_text(strip=True) if jc_span else cells[1].get_text(strip=True)
        employer_code = ec_span.get_text(strip=True) if ec_span else ""
        ad_code       = ac_span.get_text(strip=True) if ac_span else ""

        # ── url
        url = (
            f"{JOB_URL}?rid=0&ac={ad_code}&jc={job_code}&ec={employer_code}"
            if ad_code and job_code and employer_code
            else ""
        )

        # ── description  (column 4)
        description = cells[3].get_text(strip=True)

        # ── opening_date  (column 5)
        opening_date = cells[4].get_text(strip=True)

        # ── closing_date  (column 6)
        closing_date = cells[5].get_text(strip=True)

        # ── location  (column 7 — usually empty)
        location = cells[6].get_text(strip=True) if len(cells) > 6 else ""

        jobs.append({
            "job_number":      job_number,
            "title":           title,
            "company":         company,
            "job_code":        job_code,
            "employer_code":   employer_code,
            "ad_code":         ad_code,
            "description":     description,
            "opening_date":    opening_date,
            "closing_date":    closing_date,
            "location":        location,
            "url":             url,
            "scraped_at":      scraped_at,
            "functional_area": fa,
        })

    return jobs


# ──────────────────────────────────────────────
#  DISPLAY
# ──────────────────────────────────────────────
def print_table(jobs: list[dict], fa: str) -> None:
    if not jobs:
        print(f"\n⚠  No jobs found for FA={fa}. Category code check karanna.")
        return

    display = ["job_number", "title", "company", "opening_date", "closing_date"]
    headers = ["#", "Title", "Company", "Opening Date", "Closing Date"]
    rows    = [[j[c] for c in display] for j in jobs]

    print(f"\n{'─'*80}")
    print(f"  TopJobs.lk  |  FA = {fa}  |  Scraped: {jobs[0]['scraped_at']}")
    print(f"  Total vacancies: {len(jobs)}")
    print(f"{'─'*80}\n")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print()


# ──────────────────────────────────────────────
#  CSV SAVE
# ──────────────────────────────────────────────
def save_csv(jobs: list[dict], fa: str, folder: str) -> str:
    if not jobs:
        print("⚠  No data to save.")
        return ""

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"topjobs_it_jobs.csv")

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(jobs)

    print(f"✅  CSV saved → {filename}")
    print(f"    {len(jobs)} rows  |  {len(CSV_COLUMNS)} columns")
    return filename


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TopJobs.lk Vacancy Scraper → CSV")
    parser.add_argument(
        "--fa", default="SDQ",
        help="Functional Area code  (default: SDQ = Software Dev & QA)"
    )
    parser.add_argument(
        "--out", default=None,
        help="CSV save folder path  (default: GUI folder picker open vena)"
    )
    args = parser.parse_args()
    fa   = args.fa.upper()

    # ── Save folder decide karana
    if args.out:
        # --out flag dena lada folder use karana
        folder = os.path.abspath(args.out)
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder exist wenawa na: {folder}")
            sys.exit(1)
        print(f"📁  Save  වෙන location එක: {folder}")
    else:
        # GUI folder picker open karana
        print("\n📂  CSV File එක Save කරන්න අවශ්‍ය, Project එකේ Souce Folder එක Select කරන්න")
        folder = pick_save_folder()

    # ── Scrape
    print(f"\n🔍  Scraping TopJobs.lk  [FA={fa}] ...")
    soup = fetch_page(fa)
    jobs = parse_jobs(soup, fa)

    # ── Show table
    print_table(jobs, fa)

    # ── Save CSV
    save_csv(jobs, fa, folder)


if __name__ == "__main__":
    main()