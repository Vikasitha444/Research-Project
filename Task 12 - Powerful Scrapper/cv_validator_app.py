"""
CV Validator — Sri Lankan IT Market
Flask web application to analyse resumes and recommend jobs.

Bugs fixed vs original:
 - KNOWN_TECH_SKILLS was declared as an empty set() before the builder
   function existed; clean_skill_list() would silently pass all skills
   through before _build_known_tech_skills() was ever called.
 - filter_active_jobs() dropped every row whose closing_date was blank
   (i.e. "never expires" jobs).  Fixed to keep those rows.
 - enrich_csv_with_descriptions() used the DataFrame *label* index for
   the % 10 cache-flush, which is non-sequential and unreliable.
   Replaced with enumerate().
 - check_formatting_quality() could raise ValueError on an empty blocks
   list after the outer `if blocks` guard, because the margin-check was
   a separate statement with no inner guard.
 - validate_github_links() called time.sleep() inside the request
   handler, stalling the server thread under gunicorn/uwsgi.  The sleep
   is removed; callers that need rate-limiting should use a thread pool.
 - Flask session stored the full job-recommendation payload, which can
   overflow the signed-cookie limit (4 kB).  Replaced with server-side
   in-memory cache keyed by a UUID stored in the session.
 - api_validate() did not call cleanup_old_previews() — fixed.
 - calculate_dimension_scores() called a private _get_score10() helper
   that referenced 'results' by key string; if any key was missing the
   outer dict.get() fell back to {} which then returned default 0.0
   correctly, but the code was fragile.  Added explicit default chains.
 - Minor: removed the unused `temperature` parameter from
   validate_with_llm (it was accepted but the Groq client ignored it
   because the local variable shadowed the outer one).  Re-added it
   correctly as a named argument to client.chat.completions.create().
 - Minor: check_gpa_in_cv() computed text_lower twice when a bundle was
   present — now uses bundle value consistently.

Score stability fixes (v2):
 - LLM score is now deterministic: temperature fixed to 0.0 and results
   are cached by file SHA-256 hash so the same CV always gets the same
   LLM score regardless of how many times it is uploaded.
 - GitHub link validation results are also cached by (file_hash, url)
   so network flakiness cannot change the score between re-uploads.

Soft skills fixes (v3):
 - check_skills_separation() BUG 1: lines containing the word "skill"
   (e.g. "Communication Skills") were silently dropped by the filter
   `"skill" not in ll`. Filter removed — section headers are already
   caught before reaching this point.
 - check_skills_separation() BUG 2: header char limit was 30, causing
   headers like "Soft Skills & Personal Qualities" (32+ chars) to be
   missed. Limit raised to 60.
 - check_skills_separation() BUG 3: a comma-separated single line like
   "Team Work, Communication, Problem Solving" only returned the first
   canonical match. Now splits on commas/semicolons and processes each
   part independently.
 - check_skills_separation() BUG 4: generic "Skills" header without the
   word "soft" or "technical" caused a section reset. Now handled
   gracefully.
 - get_job_recommendations() now reads extracted_soft_skills from the
   validation results and incorporates a soft-skill match score (20%)
   into the final ranking. Title-based inference is used for job-side
   soft skill expectations (no extra API calls).

Job recommendation fixes (v4):
 - BUG 1: clean_skill_list() was applied to job-side skills with the
   KNOWN_TECH_SKILLS filter, dropping legitimate tools (Jenkins, Appium,
   TestNG, Maven, VMware, Illustrator, InDesign, etc.) not present in
   TECH_KEYWORD_VARIANTS. Added clean_job_skill_list() — permissive
   normalisation without the KNOWN filter — used on the job side only.
 - BUG 2: _SKILL_SYNONYMS missing multi-word LLM variants like
   "Microsoft Azure", "Agile Methodologies", "Google Cloud Platform",
   causing them to not normalise to their canonical token.
 - BUG 3: build_combined_fields() only merged inferred skills when
   job_skill_list had < 3 entries. Now always merges so every job has
   the best possible skill coverage.
 - BUG 4: _overlap_ratio() returned 0.0 when job had no skill data,
   unfairly penalising jobs with good TF-IDF scores. Now returns 0.3
   (neutral) so TF-IDF can still rank them correctly.

OCR fixes (v5):
 - CRITICAL BUG: extract_pdf_bundle() initialised EasyOCR reader but
   called pytesseract.image_to_string() instead — now uses
   reader.readtext() consistently throughout.
 - preprocess_for_ocr() was dead code (never called). Now integrated
   into extract_pdf_bundle() and scrape_job_page() for better accuracy.
 - cv2 was imported AFTER preprocess_for_ocr() used it, causing a
   potential NameError at runtime. All imports moved to the top.
 - scrape_job_page() already used EasyOCR correctly; now uses
   preprocess_for_ocr() for higher accuracy on low-quality flyer images.
 - Removed duplicate `import numpy as np` and duplicate dict keys in
   TECH_KEYWORD_VARIANTS / _SKILL_SYNONYMS.

Seniority + OCR scraper fixes (v6):
 - BUG 1: _RE_EXP_REQ regex did not handle range format "4-5 years" (the
   dash was not in the pattern) and broke on "post qualification" text
   appearing between "years" and "experience". Both fixed.
 - BUG 2: classify_job_level() defaulted to "intern_junior" for any job
   without explicit seniority keywords. Changed default to "mid" — most
   unlabelled IT engineering jobs are mid-level hires.
 - BUG 3: Senior threshold was >= 5 years; lowered to >= 4 so that
   "minimum 4-5 years" correctly classifies as senior.
 - BUG 4: scrape_job_page() only OCR-d images whose src matched a narrow
   pattern (JobAdvertismentServlet|vac-ad|flyer). TopJobs flyer images
   often live at /applicant/CVServlet or CDN paths. Replaced with a
   full-page image sweep that OCRs any content-sized image and merges
   the result into the description so Java, Spring Boot, CI/CD and
   experience requirements are captured.
 - BUG 5: _needs_enrich() skipped rows with description length >= 40
   chars, so already-scraped rows with short generic descriptions (missing
   OCR flyer content) were never re-scraped. Added a re-scrape condition
   for descriptions < 300 chars that lack experience/qualification words.

Scraper image filter fixes (v7):
 - BUG 1: scrape_job_page() was OCR-ing ../images/application.png — the
   TopJobs "Apply Now" button image — because it wasn't in the skip list
   and was > 5 KB. Added application.png / apply.png / submit.png /
   click.png and other UI button patterns to _SKIP_IMG_SRC_RE.
 - BUG 2: OCR results of 125 chars (button text level) were accepted as
   job content. Added _MIN_OCR_CONTENT_CHARS = 200 — results shorter than
   this are discarded as they contain only button/watermark text.
 - BUG 3: All images were swept in a single pass. Replaced with two-pass
   approach: Pass A tries priority images (JobAdvertismentServlet, etc.)
   first; Pass B (generic sweep) only runs if description is still short
   after Pass A + HTML extraction, and requires 10 KB minimum image size.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import pymupdf
import pymupdf4llm
import requests
from bs4 import BeautifulSoup
from flask import (
    Flask, abort, jsonify, render_template, request,
    send_from_directory, session, url_for,
)
from openai import OpenAI
from PIL import Image as _PILImage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────────────────────
# EasyOCR — initialised ONCE at startup (never inside loops)
# ──────────────────────────────────────────────────────────────
import easyocr

_OCR_AVAILABLE = False
_easyocr_reader: Optional[easyocr.Reader] = None

try:
    _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    _OCR_AVAILABLE  = True
    print("\n[SUCCESS] EasyOCR is ready!")
except Exception as _ocr_init_err:
    print(f"\n[WARNING] EasyOCR initialisation failed: {_ocr_init_err}")

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
try:
    from config import GROQ_API_KEY  # type: ignore[import]
except ImportError:
    GROQ_API_KEY: Optional[str] = None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

BASE_DIR = os.path.dirname(__file__)
PREVIEW_FOLDER = os.path.join(BASE_DIR, "previews")
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

HTTP_TIMEOUT = 10
DEFAULT_UA = {"User-Agent": "Mozilla/5.0"}
PREVIEW_TTL_SECONDS = 30 * 60

CSV_PATH  = "topjobs_it_jobs.csv"
CACHE_CSV = "topjobs_description_cache.csv"

# Server-side recommendation cache (avoids Flask session cookie overflow).
# Capped at 200 entries to prevent unbounded memory growth on busy servers.
_REC_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_REC_CACHE_MAX = 200


def _rec_cache_set(key: str, value: List[Dict[str, Any]]) -> None:
    """Insert into _REC_CACHE, evicting oldest entries when over the cap."""
    if len(_REC_CACHE) >= _REC_CACHE_MAX:
        for old_key in list(_REC_CACHE.keys())[:20]:
            _REC_CACHE.pop(old_key, None)
    _REC_CACHE[key] = value

# Score-stability caches (capped to prevent memory leaks on long-running servers)
_LLM_SCORE_CACHE:    Dict[str, Dict[str, Any]]   = {}
_GITHUB_URL_CACHE:   Dict[str, Tuple[bool, str]] = {}
_LLM_CACHE_MAX    = 500
_GITHUB_CACHE_MAX = 1000


def _llm_cache_set(key: str, value: Dict[str, Any]) -> None:
    if len(_LLM_SCORE_CACHE) >= _LLM_CACHE_MAX:
        for k in list(_LLM_SCORE_CACHE.keys())[:50]:
            _LLM_SCORE_CACHE.pop(k, None)
    _LLM_SCORE_CACHE[key] = value


def _github_cache_set(key: str, value: Tuple[bool, str]) -> None:
    if len(_GITHUB_URL_CACHE) >= _GITHUB_CACHE_MAX:
        for k in list(_GITHUB_URL_CACHE.keys())[:100]:
            _GITHUB_URL_CACHE.pop(k, None)
    _GITHUB_URL_CACHE[key] = value

# ──────────────────────────────────────────────────────────────
# SQLite
# ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "data", "scores.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash     TEXT,
                filename      TEXT,
                overall_score REAL,
                grade         TEXT,
                llm_score     REAL,
                llm_std       REAL,
                created_at    TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_resume_run(
    file_hash: str,
    filename: str,
    overall_score: float,
    grade: str,
    llm_score: float,
    llm_std: float,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO resume_runs
                (file_hash, filename, overall_score, grade, llm_score, llm_std, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_hash, filename, overall_score, grade,
                llm_score, llm_std,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


init_db()

# ──────────────────────────────────────────────────────────────
# Tech keyword variants
# (Removed duplicate keys: "Figma", "Microservices", "RabbitMQ", "Power BI")
# ──────────────────────────────────────────────────────────────
TECH_KEYWORD_VARIANTS: Dict[str, List[str]] = {
    "Java":             [r"\bjava\b(?!\s*script)"],
    "Python":           [r"\bpython\b"],
    "JavaScript":       [r"\bjavascript\b", r"\bjs\b"],
    "C#":               [r"\bc\s*#\b", r"\bcsharp\b", r"\bc\s*sharp\b"],
    "C++":              [r"\bc\+\+\b"],
    "MySQL":            [r"\bmysql\b"],
    "PostgreSQL":       [r"\bpostgres(?:ql)?\b"],
    "MongoDB":          [r"\bmongo(?:db)?\b"],
    "Oracle":           [r"\boracle\b"],
    "Git":              [r"\bgit\b"],
    "GitHub":           [r"\bgithub\b"],
    "GitLab":           [r"\bgitlab\b"],
    "HTML5":            [r"\bhtml5\b", r"\bhtml\b"],
    "CSS3":             [r"\bcss3\b", r"\bcss\b"],
    "React":            [r"\breact(?:\.js|js)?\b"],
    "Angular":          [r"\bangular\b", r"\bangularjs\b"],
    "Vue.js":           [r"\bvue(?:\.js|js)?\b"],
    "Node.js":          [r"\bnode(?:\.js|js)?\b"],
    "Express":          [r"\bexpress(?:\.js|js)?\b"],
    "Django":           [r"\bdjango\b"],
    "Flask":            [r"\bflask\b"],
    "Spring Boot":      [r"\bspring\s*boot\b", r"\bspringboot\b"],
    "Spring":           [r"\bspring\b"],
    "Android":          [r"\bandroid\b"],
    "iOS":              [r"\bios\b"],
    "Flutter":          [r"\bflutter\b"],
    "React Native":     [r"\breact\s*native\b"],
    "AWS":              [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "Azure":            [r"\bazure\b", r"\bmicrosoft\s+azure\b"],
    "GCP":              [r"\bgcp\b", r"\bgoogle\s+cloud\b", r"\bgoogle\s+cloud\s+platform\b"],
    "Docker":           [r"\bdocker\b"],
    "Kubernetes":       [r"\bkubernetes\b", r"\bk8s\b"],
    "REST API":         [r"\brest(?:ful)?\s*api(?:s)?\b"],
    "GraphQL":          [r"\bgraphql\b"],
    "Microservices":    [r"\bmicroservices?\b"],
    "Agile":            [r"\bagile\b", r"\bagile\s+methodolog(?:y|ies)\b"],
    "Scrum":            [r"\bscrum\b"],
    "JIRA":             [r"\bjira\b"],
    "TypeScript":       [r"\btypescript\b"],
    "PHP":              [r"\bphp\b"],
    "Ruby":             [r"\bruby\b"],
    "Go":               [r"\bgolang\b", r"\bgo\b"],
    "Kotlin":           [r"\bkotlin\b"],
    "Swift":            [r"\bswift\b"],
    "Scala":            [r"\bscala\b"],
    "Rust":             [r"\brust\b"],
    "Next.js":          [r"\bnext\.?js\b"],
    "Tailwind":         [r"\btailwind\b"],
    "Redux":            [r"\bredux\b"],
    "Laravel":          [r"\blaravel\b"],
    "ASP.NET":          [r"\basp\.net\b", r"\basp\s+net\b"],
    "FastAPI":          [r"\bfastapi\b"],
    "SQL":              [r"\bsql\b", r"\bt-sql\b", r"\bpl\.sql\b"],
    "Redis":            [r"\bredis\b"],
    "Elasticsearch":    [r"\belasticsearch\b"],
    "SQLite":           [r"\bsqlite\b"],
    "DynamoDB":         [r"\bdynamodb\b"],
    "Cassandra":        [r"\bcassandra\b"],
    "CI/CD":            [r"\bci/cd\b", r"\bcicd\b", r"\bjenkins\b", r"\bgithub\s+actions\b", r"\bgitlab\s+ci\b"],
    "Terraform":        [r"\bterraform\b"],
    "Linux":            [r"\blinux\b", r"\bubuntu\b", r"\bcentos\b"],
    "gRPC":             [r"\bgrpc\b"],
    "Machine Learning": [r"\bmachine\s+learning\b"],
    "Deep Learning":    [r"\bdeep\s+learning\b"],
    "TensorFlow":       [r"\btensorflow\b"],
    "PyTorch":          [r"\bpytorch\b"],
    "Pandas":           [r"\bpandas\b"],
    "NumPy":            [r"\bnumpy\b"],
    "scikit-learn":     [r"\bscikit[\-.]learn\b", r"\bsklearn\b"],
    "Spark":            [r"\bapache\s+spark\b", r"\bpyspark\b", r"\bspark\b"],
    "Hadoop":           [r"\bhadoop\b"],
    "Tableau":          [r"\btableau\b"],
    "Power BI":         [r"\bpower\s+bi\b", r"\bpowerbi\b"],
    "Selenium":         [r"\bselenium\b"],
    "JUnit":            [r"\bjunit\b"],
    "Jest":             [r"\bjest\b"],
    "Pytest":           [r"\bpytest\b"],
    "Figma":            [r"\bfigma\b"],
    "Photoshop":        [r"\bphotoshop\b"],
    "Appium":           [r"\bappium\b"],
    "Maven":            [r"\bmaven\b"],
    "Gradle":           [r"\bgradle\b"],
    "TestNG":           [r"\btestng\b"],
    "Cucumber":         [r"\bcucumber\b"],
    "Postman":          [r"\bpostman\b"],
    "Jenkins":          [r"\bjenkins\b"],
    "Ansible":          [r"\bansible\b"],
    "Nginx":            [r"\bnginx\b"],
    "RabbitMQ":         [r"\brabbitmq\b"],
    "Kafka":            [r"\bkafka\b", r"\bapache\s+kafka\b"],
    "VMware":           [r"\bvmware\b"],
    "Illustrator":      [r"\billustrator\b"],
    "InDesign":         [r"\bindesign\b"],
    "SAP":              [r"\bsap\b"],
    "Salesforce":       [r"\bsalesforce\b"],
    "SharePoint":       [r"\bsharepoint\b"],
    "ITIL":             [r"\bitil\b"],
    "OOP":              [r"\boop\b", r"\bobject[\s\-]oriented\b"],
    "Hibernate":        [r"\bhibernate\b"],
    "JPA":              [r"\bjpa\b"],
    "Swagger":          [r"\bswagger\b"],
    "OAuth":            [r"\boauth\b"],
    "JWT":              [r"\bjwt\b"],
    "ELK Stack":        [r"\belk\s*stack\b", r"\belasticsearch.*kibana\b"],
    # ── CSV-specific entries found in topjobs_it_jobs_enriched.csv ────────
    "Adobe Creative Suite": [r"\badobe\s+creative\s+suite\b", r"\badobe\s+cc\b"],
    "Adobe Photoshop":   [r"\badobe\s+photoshop\b"],
    "Adobe Illustrator": [r"\badobe\s+illustrator\b"],
    "Adobe InDesign":    [r"\badobe\s+indesign\b"],
    "Adobe After Effects":[r"\badobe\s+after\s+effects\b", r"\bafter\s+effects\b"],
    "Adobe Premiere Pro":[r"\badobe\s+premiere\b", r"\bpremiere\s+pro\b"],
    "DaVinci Resolve":   [r"\bdavinci\s+resolve\b"],
    "Sketch":            [r"\bsketch\b(?!\s*pad)"],
    "InVision":          [r"\binvision\b"],
    "UI/UX":             [r"\bui\s*/\s*ux\b", r"\bui/ux\b", r"\bux\s+design\b", r"\bui\s+design\b"],
    "Graphic Design":    [r"\bgraphic\s+design\b"],
    "Web Design":        [r"\bweb\s+design\b"],
    "Visual Design":     [r"\bvisual\s+design\b"],
    "Motion Graphics":   [r"\bmotion\s+graphics\b"],
    "Video Editing":     [r"\bvideo\s+edit(?:ing)?\b"],
    "Color Grading":     [r"\bcolor\s+grad(?:ing)?\b", r"\bcolour\s+grad(?:ing)?\b"],
    "Microsoft Office":  [r"\bmicrosoft\s+office\b", r"\bms\s+office\b"],
    "Microsoft Project": [r"\bmicrosoft\s+project\b", r"\bms\s+project\b"],
    "Microsoft Office 365": [r"\boffice\s+365\b", r"\bm365\b", r"\bmicrosoft\s+365\b"],
    "Microsoft Dynamics":[r"\bmicrosoft\s+dynamics\b", r"\bms\s+dynamics\b"],
    "PowerShell":        [r"\bpowershell\b"],
    "Active Directory":  [r"\bactive\s+directory\b", r"\bldap\b"],
    "Windows Server":    [r"\bwindows\s+server\b"],
    "Visio":             [r"\bvisio\b"],
    "Excel":             [r"\bmicrosoft\s+excel\b", r"\bms\s+excel\b", r"\bexcel\b"],
    "Visual Studio":     [r"\bvisual\s+studio\b"],
    "Entity Framework":  [r"\bentity\s+framework\b"],
    ".NET Core":         [r"\b\.net\s+core\b", r"\bdotnet\s+core\b"],
    "Cyber Security":    [r"\bcyber\s*security\b", r"\binformation\s+security\b", r"\binfosec\b"],
    "Networking":        [r"\bnetworking\b", r"\bnetwork\s+administration\b", r"\bcisco\s+networking\b"],
    "Networking Fundamentals": [r"\bnetworking\s+fundamentals\b"],
    "Windows":           [r"\bwindows\b"],
    "Virtualization":    [r"\bvirtualization\b", r"\bhyper-v\b"],
    "Incident Response": [r"\bincident\s+response\b"],
    "CompTIA Security+": [r"\bcomptia\s+security\+?\b", r"\bsecurity\+\b"],
    "CISSP":             [r"\bcissp\b"],
    "Google Cloud":      [r"\bgoogle\s+cloud\b", r"\bgcp\b"],
    "Cloud Security":    [r"\bcloud\s+security\b"],
    "Prometheus":        [r"\bprometheus\b"],
    "Grafana":           [r"\bgrafana\b"],
    "Data Visualization":[r"\bdata\s+visuali[sz]ation\b"],
    "Data Modeling":     [r"\bdata\s+modeling\b", r"\bdata\s+modelling\b"],
    "Data Migration":    [r"\bdata\s+migration\b"],
    "Data Mining":       [r"\bdata\s+mining\b"],
    "Data Warehousing":  [r"\bdata\s+warehous(?:ing)?\b"],
    "ETL":               [r"\betl\b", r"\betl\s+tools\b"],
    "Business Intelligence": [r"\bbusiness\s+intelligence\b", r"\b\bbi\b"],
    "Google Analytics":  [r"\bgoogle\s+analytics\b"],
    "SEO":               [r"\bseo\b", r"\bsearch\s+engine\s+optim\w+\b"],
    "Dart":              [r"\bdart\b"],
    "Firebase":          [r"\bfirebase\b"],
    "HTML/CSS":          [r"\bhtml\s*/\s*css\b", r"\bhtml/css\b"],
    "Trello":            [r"\btrello\b"],
    "Asana":             [r"\basana\b"],
    "ABAP":              [r"\babap\b"],
    "SAP HANA":          [r"\bsap\s+hana\b"],
    "SAP S/4HANA":       [r"\bsap\s+s\s*/\s*4\s*hana\b"],
    "SAP ERP":           [r"\bsap\s+erp\b"],
    "Digital Marketing": [r"\bdigital\s+marketing\b"],
    "Social Media Marketing": [r"\bsocial\s+media\s+marketing\b"],
    "Content Creation":  [r"\bcontent\s+creat(?:ion|e|or)\b"],
    "WordPress":         [r"\bwordpress\b"],
    "Troubleshooting":   [r"\btroubleshoot(?:ing)?\b"],
    "IT Project Management": [r"\bit\s+project\s+manag\w+\b"],
    "IT Service Management": [r"\bit\s+service\s+manag\w+\b", r"\bitsm\b"],
    "ITIL Foundation":   [r"\bitil\s+foundation\b"],
    "Project Management":[r"\bproject\s+manag\w+\b(?!\s+professional)"],
    "Business Analysis": [r"\bbusiness\s+analy(?:sis|st)\b"],
    "Requirements Gathering": [r"\brequirements?\s+gather\w+\b"],
    "Stakeholder Management": [r"\bstakeholder\s+manag\w+\b"],
    "Microservices Architecture": [r"\bmicroservices?\s+architect\w+\b"],
    "ERP":               [r"\berp\b", r"\berp\s+implementation\b"],
    "Process Improvement":[r"\bprocess\s+improvement\b"],
    "Financial Accounting":[r"\bfinancial\s+account\w+\b"],
    "Photography":       [r"\bphotography\b"],
    "Sound Design":      [r"\bsound\s+design\b"],
    "Branding":          [r"\bbranding\b"],
    "Typography":        [r"\btypography\b"],
    "Color Theory":      [r"\bcolou?r\s+theory\b"],
    # Observability / monitoring (v6 — from job flyer content)
    "Observability":     [r"\bobservabilit\w+\b"],
    "Monitoring":        [r"\bmonitoring\b"],
    "On-Call":           [r"\bon[\s\-]?call\b"],
}

SRI_LANKAN_TECH_KEYWORDS: List[str] = [
    "Java", "Python", "JavaScript", "C#", "C++", "MySQL", "PostgreSQL", "MongoDB",
    "Oracle", "Git", "GitHub", "GitLab", "HTML5", "CSS3", "React", "Angular",
    "Vue.js", "Node.js", "Express", "Django", "Flask", "Spring Boot", "Android",
    "iOS", "Flutter", "React Native", "AWS", "Azure", "Docker", "Kubernetes",
    "REST API", "GraphQL", "Microservices", "Agile", "Scrum", "JIRA",
]

# ──────────────────────────────────────────────────────────────
# Title normalisation patterns (ordered longest-match first)
# ──────────────────────────────────────────────────────────────
TITLE_ROLE_PATTERNS: List[Tuple[str, str]] = [
    ("full stack",            "fullstack"),
    ("full-stack",            "fullstack"),
    ("react native",          "react_native"),
    ("machine learning",      "ml"),
    ("deep learning",         "deep_learning"),
    ("data engineer",         "data_engineer"),
    ("data scientist",        "data_scientist"),
    ("data analyst",          "data_analyst"),
    ("business intelligence", "bi"),
    ("business analyst",      "business_analyst"),
    ("quality assurance",     "qa"),
    ("test automation",       "qa_automation"),
    ("system admin",          "sysadmin"),
    ("it support",            "it_support"),
    ("it officer",            "it"),
    ("it executive",          "it"),
    ("network engineer",      "network"),
    ("network admin",         "network"),
    ("cyber security",        "security"),
    ("information security",  "security"),
    ("ui/ux",                 "ui_ux"),
    ("ui ux",                 "ui_ux"),
    ("graphic design",        "graphic"),
    ("packaging design",      "packaging"),
    ("visual design",         "visual_design"),
    ("spring boot",           "spring_boot"),
    ("power bi",              "power_bi"),
    ("asp.net",               "dotnet"),
    ("react",       "react"),
    ("angular",     "angular"),
    ("vue",         "vue"),
    ("flutter",     "flutter"),
    ("android",     "android"),
    ("ios",         "ios"),
    ("kotlin",      "kotlin"),
    ("swift",       "swift"),
    ("node",        "node"),
    ("express",     "express"),
    ("python",      "python"),
    ("django",      "django"),
    ("flask",       "flask"),
    ("fastapi",     "fastapi"),
    ("java",        "java"),
    ("spring",      "spring"),
    ("php",         "php"),
    ("laravel",     "laravel"),
    ("dotnet",      "dotnet"),
    (".net",        "dotnet"),
    ("devops",      "devops"),
    ("cloud",       "cloud"),
    ("aws",         "aws"),
    ("azure",       "azure"),
    ("gcp",         "gcp"),
    ("kubernetes",  "kubernetes"),
    ("docker",      "docker"),
    ("terraform",   "terraform"),
    ("ansible",     "ansible"),
    ("kafka",       "kafka"),
    ("spark",       "spark"),
    ("hadoop",      "hadoop"),
    ("tableau",     "tableau"),
    ("salesforce",  "salesforce"),
    ("sharepoint",  "sharepoint"),
    ("wordpress",   "wordpress"),
    ("sap",         "sap"),
    ("selenium",    "qa_automation"),
    ("automation",  "qa_automation"),
    ("testing",     "qa"),
    ("security",    "security"),
    ("network",     "network"),
    ("mobile",      "mobile"),
    ("frontend",    "frontend"),
    ("front-end",   "frontend"),
    ("backend",     "backend"),
    ("back-end",    "backend"),
    ("database",    "database"),
    ("dba",         "database"),
    ("golang",      "golang"),
    ("rust",        "rust"),
    ("scala",       "scala"),
    ("ruby",        "ruby"),
    ("typescript",  "typescript"),
    ("next",        "nextjs"),
    ("nuxt",        "nuxt"),
    ("graphql",     "graphql"),
    ("microservice","microservices"),
    ("ui",          "ui"),
    ("ux",          "ux"),
    ("graphic",     "graphic"),
    ("design",      "designer"),
    ("packaging",   "packaging"),
    ("helpdesk",    "it_support"),
    ("support",     "it_support"),
    ("qa",          "qa"),
    ("erp",         "erp"),
    ("crm",         "crm"),
    ("intern",      "intern"),
    ("trainee",     "trainee"),
    ("graduate",    "graduate"),
    ("placement",   "graduate"),
    ("consultant",  "consultant"),
    ("analyst",     "analyst"),
    ("architect",   "architect"),
    ("manager",     "manager"),
    ("lead",        "lead"),
]


def normalize_title_to_role(title: str) -> str:
    t = (title or "").lower().strip()
    for pattern, role_key in TITLE_ROLE_PATTERNS:
        if pattern in t:
            return role_key
    return "general"


def _extract_skills_for_data_map(raw_skills: str, description: str = "") -> List[str]:
    combined     = (raw_skills or "") + " " + (description or "")
    regex_skills = extract_skills_from_text(combined)
    regex_set    = set(regex_skills)

    _GENERIC_WORDS = {
        "intern", "trainee", "junior", "senior", "lead", "associate",
        "graduate", "engineer", "developer", "manager", "analyst",
        "consultant", "officer", "required", "skills", "experience",
        "years", "strong", "knowledge", "good", "ability", "excellent",
    }
    split_skills: List[str] = []
    for part in re.split(r"[,|\n]+", raw_skills or ""):
        part = part.strip()
        if not part or len(part) < 2:
            continue
        ns = normalize_token(part)
        if ns not in _GENERIC_WORDS and len(ns) >= 2 and ns not in regex_set:
            split_skills.append(ns)
            regex_set.add(ns)

    return list(dict.fromkeys(regex_skills + split_skills))


# ──────────────────────────────────────────────────────────────
# Data-driven title → skill inference
# ──────────────────────────────────────────────────────────────
_DATA_SKILL_MAP: Dict[str, List[str]] = {}
_DATA_SKILL_MAP_BUILT: bool = False

_STATIC_SKILL_FALLBACK: Dict[str, List[str]] = {
    "general":      ["python", "sql", "git", "rest api"],
    "intern":       ["python", "sql", "git", "html5", "css3"],
    "trainee":      ["python", "sql", "git"],
    "graduate":     ["python", "sql", "git", "rest api"],
    "consultant":   ["sql", "python", "rest api", "git"],
    "manager":      ["sql", "python", "git", "docker"],
    "lead":         ["python", "sql", "docker", "git", "rest api"],
    "architect":    ["python", "java", "docker", "kubernetes", "rest api"],
    "analyst":      ["sql", "python", "pandas", "power bi"],
}


def build_data_skill_map(
    df: pd.DataFrame,
    top_n: int = 10,
    min_freq: int = 1,
) -> Dict[str, List[str]]:
    from collections import Counter, defaultdict

    role_counter: Dict[str, Counter] = defaultdict(Counter)

    for _, row in df.iterrows():
        title = str(row.get("title",       "") or "")
        raw   = str(row.get("raw_skills",  "") or "")
        desc  = str(row.get("description", "") or "")

        if not raw.strip() and len(desc.strip()) < 30:
            continue

        role_key = normalize_title_to_role(title)
        skills   = _extract_skills_for_data_map(raw, desc)
        if not skills:
            continue

        for skill in skills:
            role_counter[role_key][skill] += 1

    result: Dict[str, List[str]] = {}
    for role_key, counter in role_counter.items():
        top = [s for s, _ in counter.most_common(top_n) if _ >= min_freq]
        if top:
            result[role_key] = top

    logger.info(f"Data-driven skill map built: {len(result)} role buckets from {len(df)} jobs.")
    return result


def infer_skills_from_title(title: str) -> List[str]:
    role_key = normalize_title_to_role(title or "")
    skills   = _DATA_SKILL_MAP.get(role_key)
    if skills:
        return skills
    parent = role_key.split("_")[0]
    skills = _DATA_SKILL_MAP.get(parent)
    if skills:
        return skills
    return list(dict.fromkeys(
        normalize_token(s)
        for s in _STATIC_SKILL_FALLBACK.get(role_key,
                 _STATIC_SKILL_FALLBACK.get(parent,
                 _STATIC_SKILL_FALLBACK["general"]))
    ))


# ──────────────────────────────────────────────────────────────
# Title → SOFT skill inference map
# ──────────────────────────────────────────────────────────────
TITLE_SOFT_SKILL_MAP: Dict[str, List[str]] = {
    "qa engineer":          ["attention_to_detail", "analytical_thinking", "communication", "teamwork", "problem_solving"],
    "quality assurance":    ["attention_to_detail", "analytical_thinking", "communication", "problem_solving"],
    "test engineer":        ["attention_to_detail", "analytical_thinking", "problem_solving", "teamwork"],
    "automation engineer":  ["analytical_thinking", "problem_solving", "attention_to_detail", "self_motivation"],
    "manager":              ["leadership", "communication", "conflict_resolution", "time_management", "mentoring"],
    "team lead":            ["leadership", "communication", "mentoring", "conflict_resolution", "teamwork"],
    "project manager":      ["leadership", "communication", "time_management", "project_management_mindset", "conflict_resolution"],
    "product manager":      ["communication", "leadership", "analytical_thinking", "presentation_skills", "customer_focus"],
    "director":             ["leadership", "communication", "conflict_resolution", "presentation_skills", "mentoring"],
    "head of":              ["leadership", "communication", "mentoring", "conflict_resolution", "time_management"],
    "assistant manager":    ["leadership", "communication", "time_management", "teamwork", "conflict_resolution"],
    "software engineer":    ["problem_solving", "teamwork", "communication", "analytical_thinking", "adaptability"],
    "software developer":   ["problem_solving", "teamwork", "communication", "analytical_thinking", "self_motivation"],
    "full stack":           ["problem_solving", "adaptability", "teamwork", "self_motivation", "communication"],
    "backend":              ["problem_solving", "analytical_thinking", "attention_to_detail", "teamwork"],
    "frontend":             ["creativity", "attention_to_detail", "communication", "adaptability", "teamwork"],
    "mobile developer":     ["problem_solving", "adaptability", "self_motivation", "attention_to_detail"],
    "android":              ["problem_solving", "adaptability", "attention_to_detail", "self_motivation"],
    "ios":                  ["problem_solving", "adaptability", "attention_to_detail", "self_motivation"],
    "data scientist":       ["analytical_thinking", "problem_solving", "communication", "attention_to_detail", "creativity"],
    "data engineer":        ["analytical_thinking", "problem_solving", "attention_to_detail", "teamwork"],
    "data analyst":         ["analytical_thinking", "attention_to_detail", "communication", "problem_solving"],
    "machine learning":     ["analytical_thinking", "problem_solving", "self_motivation", "creativity"],
    "ai engineer":          ["analytical_thinking", "problem_solving", "creativity", "self_motivation"],
    "business intelligence":["analytical_thinking", "communication", "attention_to_detail", "presentation_skills"],
    "devops":               ["problem_solving", "adaptability", "teamwork", "communication", "analytical_thinking"],
    "cloud engineer":       ["problem_solving", "adaptability", "analytical_thinking", "self_motivation"],
    "system administrator": ["problem_solving", "analytical_thinking", "attention_to_detail", "communication"],
    "network":              ["problem_solving", "analytical_thinking", "attention_to_detail", "communication"],
    "security":             ["analytical_thinking", "attention_to_detail", "problem_solving", "communication"],
    "ui designer":          ["creativity", "communication", "attention_to_detail", "adaptability", "customer_focus"],
    "ux designer":          ["creativity", "communication", "customer_focus", "analytical_thinking", "attention_to_detail"],
    "graphic designer":     ["creativity", "attention_to_detail", "communication", "adaptability", "time_management"],
    "packaging designer":   ["creativity", "attention_to_detail", "communication", "time_management"],
    "visual designer":      ["creativity", "attention_to_detail", "communication", "adaptability"],
    "it support":           ["communication", "problem_solving", "customer_focus", "adaptability"],
    "helpdesk":             ["communication", "customer_focus", "problem_solving"],
    "it officer":           ["communication", "problem_solving", "teamwork", "adaptability"],
    "it executive":         ["communication", "problem_solving", "leadership", "teamwork"],
    "business analyst":     ["analytical_thinking", "communication", "presentation_skills", "problem_solving", "customer_focus"],
    "systems analyst":      ["analytical_thinking", "communication", "attention_to_detail", "problem_solving"],
    "intern":               ["adaptability", "self_motivation", "teamwork", "communication", "time_management"],
    "trainee":              ["adaptability", "self_motivation", "teamwork", "communication"],
    "associate":            ["teamwork", "communication", "adaptability", "self_motivation"],
    "graduate":             ["adaptability", "self_motivation", "teamwork", "communication"],
    "junior":               ["adaptability", "self_motivation", "teamwork", "problem_solving"],
    "entry level":          ["adaptability", "self_motivation", "teamwork", "communication"],
    "consultant":           ["communication", "analytical_thinking", "problem_solving", "presentation_skills", "customer_focus"],
}

_DEFAULT_SOFT_SKILLS: List[str] = [
    "communication", "teamwork", "problem_solving", "adaptability", "time_management",
]

_CV_SECTION_SOFT_PATTERNS: Dict[str, List[str]] = {
    "communication": [
        r"\bcommunicat\w*",
        r"\bverbal\b",
        r"\bwritten\s+(?:skill|communication|english)",
        r"\bpresentation\s+skill",
        r"\bpublic\s+speaking",
        r"\binterpersonal",
        r"\barticulate\b",
        r"\bactive\s+listen\w+",
    ],
    "teamwork": [
        r"\bteam[\s\-]?work",
        r"\bteam\s+player",
        r"\bcollaborat\w+",
        r"\bco[\s\-]?operat\w+",
        r"\bgroup\s+work",
        r"\bcross[\s\-]?functional",
        r"\bwork\w*\s+(?:with|alongside)\s+team",
    ],
    "problem_solving": [
        r"\bproblem[\s\-]?solv\w+",
        r"\btroubleshoot\w+",
        r"\bcritical\s+think\w+",
        r"\blogical\s+think\w+",
        r"\bdiagnos\w+",
        r"\bdebugg?\w+",
    ],
    "leadership": [
        r"\bleadership",
        r"\bteam\s+lead",
        r"\bmentor\w*",
        r"\bcoach\w*",
        r"\bguide\s+(?:and\s+)?(?:junior|team)",
        r"\bmanage\s+(?:and\s+)?(?:lead|team|junior)",
    ],
    "time_management": [
        r"\btime[\s\-]?management",
        r"\bdeadline",
        r"\bprioritiz\w+",
        r"\bmulti[\s\-]?task\w+",
        r"\borganiz\w+",
        r"\bschedul\w+",
        r"\bmeet\s+deadline",
        r"\bwork\w*\s+under\s+pressure",
    ],
    "adaptability": [
        r"\badaptab\w+",
        r"\bflexib\w+",
        r"\bquick\s+learn\w+",
        r"\bfast\s+learn\w+",
        r"\bopen\s+to\s+change",
        r"\blearn\w+\s+new\s+(?:skill|technolog)",
        r"\bdynamic\s+environment",
        r"\bfast[\s\-]?paced",
    ],
    "creativity": [
        r"\bcreat(?:ive|ivity)",
        r"\binnovat\w+",
        r"\bdesign\s+think\w+",
        r"\bout[\s\-]?of[\s\-]?the[\s\-]?box",
        r"\boriginal\s+(?:idea|design|concept)",
    ],
    "attention_to_detail": [
        r"\battention\s+to\s+detail",
        r"\bdetail[\s\-]?oriented",
        r"\baccurate\b",
        r"\bprecise\b",
        r"\bmeticulous",
        r"\bquality[\s\-]?conscious",
        r"\bhigh[\s\-]?quality",
    ],
    "customer_focus": [
        r"\bcustomer[\s\-]?(?:focus|service|centric|satisf)",
        r"\bclient[\s\-]?(?:focus|facing|relation|satisf)",
        r"\bend[\s\-]?user",
        r"\buser\s+(?:experience|satisfaction|need)",
    ],
    "self_motivation": [
        r"\bself[\s\-]?motivat\w+",
        r"\bproactiv\w+",
        r"\binitiativ\w+",
        r"\bself[\s\-]?driven",
        r"\bindependent\s+work",
        r"\bwork\w*\s+independently",
        r"\bself[\s\-]?starter",
        r"\beager\s+to\s+learn",
        r"\bhighly\s+motivated",
    ],
    "analytical_thinking": [
        r"\banalytical\b",
        r"\bdata[\s\-]?driven",
        r"\bquantitative",
        r"\bresearch\s+skill",
        r"\banalyz\w+\s+(?:data|requirement|problem)",
        r"\bsystematic\b",
    ],
    "conflict_resolution": [
        r"\bconflict\s+(?:resolution|management)",
        r"\bnegotiat\w+",
        r"\bmediat\w+",
    ],
    "mentoring": [
        r"\bmentor\w*",
        r"\bcoach\w*",
        r"\bknowledge\s+shar\w+",
        r"\btrain\w+\s+(?:junior|team|staff)",
    ],
    "presentation_skills": [
        r"\bpresentation\b",
        r"\bpublic\s+speaking",
        r"\bpitch\w*",
        r"\bpresent\w+\s+to\s+(?:stakeholder|client|management)",
    ],
    "project_management_mindset": [
        r"\bproject\s+manag\w+",
        r"\bscrum\s+master",
        r"\bproject\s+plan\w+",
        r"\bdeliver\w+\s+project",
    ],
}

# ──────────────────────────────────────────────────────────────
# Skill normalisation
# ──────────────────────────────────────────────────────────────
_SKILL_SYNONYMS: Dict[str, str] = {
    "js":                           "javascript",
    "reactjs":                      "react",
    "react.js":                     "react",
    "angular.js":                   "angular",
    "angularjs":                    "angular",
    "vue.js":                       "vue.js",
    "vuejs":                        "vue.js",
    "next.js":                      "next.js",
    "nextjs":                       "next.js",
    "node":                         "node.js",
    "nodejs":                       "node.js",
    "expressjs":                    "express",
    "express.js":                   "express",
    "express js":                   "express",
    "jquery":                       "javascript",
    "ajax":                         "javascript",
    "bootstrap":                    "css3",
    "html/css":                     "html5",
    "html css":                     "html5",
    "restapi":                      "rest api",
    "rest":                         "rest api",
    "restful":                      "rest api",
    "restful api":                  "rest api",
    "restful apis":                 "rest api",
    "rest apis":                    "rest api",
    "rest api design":              "rest api",
    "api design":                   "rest api",
    "api testing":                  "rest api",
    "postgres":                     "postgresql",
    "mysql":                        "sql",
    "my sql":                       "sql",
    "mysql database":               "sql",
    "mongo":                        "mongodb",
    "nosql":                        "mongodb",
    "nosql databases":              "mongodb",
    "sql server":                   "sql",
    "database management":          "sql",
    "sql database":                 "sql",
    "sql databases":                "sql",
    "relational database":          "sql",
    "data warehousing":             "sql",
    "data modeling":                "sql",
    "data modelling":               "sql",
    "data migration":               "sql",
    "etl tools":                    "etl",
    "etl":                          "etl",
    "microsoft azure":              "azure",
    "ms azure":                     "azure",
    "google cloud":                 "gcp",
    "google cloud platform":        "gcp",
    "amazon web services":          "aws",
    "cloud computing":              "aws",
    "cloud security":               "aws",
    "devops":                       "ci/cd",
    "ci/cd pipelines":              "ci/cd",
    "ci cd":                        "ci/cd",
    "github actions":               "ci/cd",
    "gitlab ci":                    "ci/cd",
    "k8s":                          "kubernetes",
    "springboot":                   "spring boot",
    "codeigniter":                  "php",
    "asp.net":                      "asp.net",
    "entity framework":             "asp.net",
    ".net":                         "asp.net",
    ".net core":                    "asp.net",
    "visual studio":                "asp.net",
    "ml":                           "machine learning",
    "dl":                           "deep learning",
    "tf":                           "tensorflow",
    "data analysis":                "sql",
    "data analytics":               "pandas",
    "data visualization":           "tableau",
    "data mining":                  "pandas",
    "business intelligence":        "tableau",
    "object oriented":              "oop",
    "object-oriented":              "oop",
    "object oriented programming":  "oop",
    "object-oriented programming":  "oop",
    "adobe photoshop":              "photoshop",
    "adobe illustrator":            "illustrator",
    "adobe indesign":               "indesign",
    "invision":                     "figma",
    "sketch":                       "figma",
    "ui/ux":                        "ui/ux",
    "ui/ux design":                 "ui/ux",
    "ux design":                    "ui/ux",
    "ui design":                    "ui/ux",
    "web design":                   "ui/ux",
    "branding":                     "graphic design",
    "typography":                   "graphic design",
    "color theory":                 "graphic design",
    "colour theory":                "graphic design",
    "print design":                 "graphic design",
    "visual design":                "graphic design",
    "digital illustration":         "illustrator",
    "after effects":                "adobe after effects",
    "premiere pro":                 "adobe premiere pro",
    "motion graphics":              "adobe after effects",
    "video editing":                "video editing",
    "color grading":                "video editing",
    "colour grading":               "video editing",
    "color correction":             "video editing",
    "davinci resolve":              "video editing",
    "final cut pro":                "video editing",
    "agile methodologies":          "agile",
    "agile methodology":            "agile",
    "agile development":            "agile",
    "agile scrum":                  "agile",
    "agile methods":                "agile",
    "cyber security":               "cyber security",
    "cybersecurity":                "cyber security",
    "information security":         "cyber security",
    "infosec":                      "cyber security",
    "comptia security+":            "cyber security",
    "cissp":                        "cyber security",
    "incident response":            "cyber security",
    "networking":                   "networking",
    "networking fundamentals":      "networking",
    "cisco networking":             "networking",
    "network administration":       "networking",
    "linux administration":         "linux",
    "linux admin":                  "linux",
    "windows server":               "windows server",
    "active directory":             "active directory",
    "virtualization":               "virtualization",
    "powershell":                   "linux",
    "it project management":        "project management",
    "it service management":        "itil",
    "itsm":                         "itil",
    "itil foundation":              "itil",
    "requirements gathering":       "business analysis",
    "stakeholder management":       "business analysis",
    "business process improvement": "business analysis",
    "process improvement":          "business analysis",
    "microservices architecture":   "microservices",
    "erp implementation":           "erp",
    "sap erp":                      "sap",
    "sap hana":                     "sap",
    "sap s/4hana":                  "sap",
    "microsoft office":             "microsoft office",
    "ms office":                    "microsoft office",
    "microsoft project":            "project management",
    "office 365":                   "microsoft office",
    "microsoft 365":                "microsoft office",
    "microsoft office 365":         "microsoft office",
    "microsoft dynamics":           "erp",
    "trello":                       "jira",
    "asana":                        "jira",
    "abap":                         "abap",
    "financial accounting":         "erp",
    "social media marketing":       "digital marketing",
    "content creation":             "digital marketing",
    "seo":                          "digital marketing",
    "google analytics":             "digital marketing",
    "digital media":                "digital marketing",
    "dart":                         "flutter",
    "firebase":                     "firebase",
    "python scripting":             "python",
    "java programming":             "java",
    "apache kafka":                 "kafka",
    "apache spark":                 "spark",
    "troubleshooting":              "troubleshooting",
    "photography":                  "photography",
    "sound design":                 "sound design",
    "wordpress":                    "wordpress",
    # v6 additions
    "observability":                "observability",
    "on-call":                      "on-call",
    "on call":                      "on-call",
    "incident management":          "incident response",
    "production support":           "troubleshooting",
    "cloud platforms":              "aws",
    "modern frontend":              "javascript",
    "frontend technologies":        "javascript",
}

_GENERIC_EMPLOYMENT_WORDS: Set[str] = {
    "intern", "internship", "trainee", "entry level", "entry-level",
    "junior", "associate", "senior", "lead", "mid", "mid-level",
    "graduate", "undergraduate", "student", "fresher", "probationary",
    "full time", "full-time", "part time", "part-time",
    "contract", "temporary", "permanent",
    "tools", "frameworks", "methodologies", "methodology",
    "testing", "tracking", "management", "development",
    "administration", "programming", "engineering", "concepts",
    "principles", "techniques", "practices", "fundamentals",
    "test automation frameworks", "defect tracking tools",
    "automation frameworks", "scripting languages",
}


def normalize_token(token: str) -> str:
    t = re.sub(r"\s+", " ", (token or "").strip().lower())
    return _SKILL_SYNONYMS.get(t, t)


def normalize_keywords(items: List[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for k in items or []:
        nk = normalize_token(k)
        if nk:
            seen[nk] = None
    return list(seen)


def _build_known_tech_skills() -> Set[str]:
    return {normalize_token(label) for label in TECH_KEYWORD_VARIANTS}


KNOWN_TECH_SKILLS: Set[str] = _build_known_tech_skills()


def extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    padded = " " + re.sub(r"\s+", " ", text) + " "
    found: List[str] = []
    for label, patterns in TECH_KEYWORD_VARIANTS.items():
        for pat in patterns:
            if re.search(pat, padded, flags=re.IGNORECASE):
                found.append(normalize_token(label))
                break
    return list(dict.fromkeys(found))


extract_job_skills = extract_skills_from_text


def clean_skill_list(skills: Any) -> List[str]:
    """CV-side skill cleaner — applies KNOWN_TECH_SKILLS whitelist."""
    if not skills:
        return []
    if isinstance(skills, str):
        parts  = re.split(r"[,|/•\n]+", skills)
        skills = [p.strip() for p in parts if p.strip()]

    out: List[str] = []
    for s in skills:
        ns = normalize_token(str(s))
        if not ns or len(ns) < 2:
            continue
        if ns in _GENERIC_EMPLOYMENT_WORDS:
            continue
        if ns not in KNOWN_TECH_SKILLS:
            continue
        out.append(ns)
    return list(dict.fromkeys(out))


def clean_job_skill_list(skills: Any) -> List[str]:
    """Job-side skill cleaner — permissive normalisation without KNOWN_TECH_SKILLS filter."""
    if not skills:
        return []

    if isinstance(skills, str):
        _SLASH_PROTECT = {
            r"\bci/cd\b":           "CI_CD_TOKEN",
            r"\bhtml/css\b":        "HTMLCSS_TOKEN",
            r"\bui/ux\b":           "UIUX_TOKEN",
            r"\bsap\s*fi/co\b":     "SAPFICO_TOKEN",
            r"\bsap\s*bw/4hana\b":  "SAPBW4HANA_TOKEN",
            r"\basp\.net\b":        "ASPNET_TOKEN",
            r"\bc\+\+\b":           "CPP_TOKEN",
            r"\bc#\b":              "CSHARP_TOKEN",
            r"\bnode\.js\b":        "NODEJS_TOKEN",
            r"\bvue\.js\b":         "VUEJS_TOKEN",
            r"\bnext\.js\b":        "NEXTJS_TOKEN",
            r"\b\.net\s+core\b":    "DOTNETCORE_TOKEN",
            r"\b\.net\b":           "DOTNET_TOKEN",
        }
        protected = skills
        for pat, tok in _SLASH_PROTECT.items():
            protected = re.sub(pat, tok, protected, flags=re.IGNORECASE)

        parts = re.split(r"[,|•\n]+", protected)

        _RESTORE = {
            "CI_CD_TOKEN":       "ci/cd",
            "HTMLCSS_TOKEN":     "html/css",
            "UIUX_TOKEN":        "ui/ux",
            "SAPFICO_TOKEN":     "sap fi/co",
            "SAPBW4HANA_TOKEN":  "sap bw/4hana",
            "ASPNET_TOKEN":      "asp.net",
            "CPP_TOKEN":         "c++",
            "CSHARP_TOKEN":      "c#",
            "NODEJS_TOKEN":      "node.js",
            "VUEJS_TOKEN":       "vue.js",
            "NEXTJS_TOKEN":      "next.js",
            "DOTNETCORE_TOKEN":  ".net core",
            "DOTNET_TOKEN":      ".net",
        }
        skills = []
        for p in parts:
            p = p.strip()
            for tok, orig in _RESTORE.items():
                p = p.replace(tok, orig)
            if p:
                skills.append(p)

    out: List[str] = []
    for s in skills:
        ns = normalize_token(str(s))
        if not ns or len(ns) < 2:
            continue
        if ns in _GENERIC_EMPLOYMENT_WORDS:
            continue
        if len(ns.split()) > 4:
            continue
        out.append(ns)
    return list(dict.fromkeys(out))


# ──────────────────────────────────────────────────────────────
# Soft skill helpers
# ──────────────────────────────────────────────────────────────

def infer_soft_skills_from_title(title: str) -> List[str]:
    if not title:
        return list(_DEFAULT_SOFT_SKILLS)
    title_lower = title.lower().strip()
    for keyword in sorted(TITLE_SOFT_SKILL_MAP.keys(), key=len, reverse=True):
        if keyword in title_lower:
            seen: Dict[str, None] = {}
            for s in TITLE_SOFT_SKILL_MAP[keyword]:
                seen[s] = None
            return list(seen)
    return list(_DEFAULT_SOFT_SKILLS)


def _clean_skill_line(line: str) -> str:
    return re.sub(r"^[\s•\-\*◦▪→✓✔►\d\.]+\s*", "", line.strip()).strip()


def _split_skill_line(line: str) -> List[str]:
    parts = re.split(r"[,;|/]+", line)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def _match_soft_skill_line(line: str) -> Optional[str]:
    for skill_name, patterns in _CV_SECTION_SOFT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, line, flags=re.IGNORECASE):
                return skill_name
    return None


def _extract_from_line(raw_line: str) -> List[str]:
    cleaned = _clean_skill_line(raw_line)
    parts   = _split_skill_line(cleaned)
    found: List[str] = []
    for part in parts:
        canonical = _match_soft_skill_line(part)
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def compute_soft_skill_match(
    cv_soft_skills: List[str],
    job_title: str,
) -> Dict[str, Any]:
    job_expected: List[str] = infer_soft_skills_from_title(job_title)
    cv_set:  Set[str] = set(cv_soft_skills)
    exp_set: Set[str] = set(job_expected)
    matched = sorted(cv_set & exp_set)
    missing = sorted(exp_set - cv_set)
    union   = cv_set | exp_set
    score   = round(len(matched) / max(len(union), 1), 3)
    return {
        "job_expected":  job_expected,
        "matched":       matched,
        "missing":       missing,
        "score":         score,
        "score_display": f"{len(matched)}/{len(job_expected)}",
    }


# ──────────────────────────────────────────────────────────────
# Seniority helpers
# ──────────────────────────────────────────────────────────────
_RE_SENIOR = re.compile(
    r"\b(senior|sr\.?|lead|manager|head|architect|principal"
    r"|tech\s+lead|team\s+lead|director|vp|vice\s+president"
    r"|cto|cio|chief)\b",
    re.IGNORECASE,
)
_RE_INTERN = re.compile(r"\b(intern(?:ship)?)\b", re.IGNORECASE)
_RE_JUNIOR = re.compile(
    r"\b(trainee|undergraduate|student|fresher|entry[\s\-]*level"
    r"|junior|jr\.?|associate|graduate|probationary"
    r"|it\s+graduate|software\s+graduate|it\s+trainee"
    r"|software\s+trainee|it\s+associate|software\s+associate)\b",
    re.IGNORECASE,
)
_RE_YEARS     = re.compile(r"(\d+)\+?\s*years?\b", re.IGNORECASE)

# ── v6 FIX: Added [-–] range support and "post qualification" phrase ──────────
_RE_EXP_REQ = re.compile(
    r"(?:"
    r"(?:minimum|at\s+least|more\s+than|over"
    r"|should\s+have|must\s+have|require[sd]?)?\s*"
    r"(\d+)\s*(?:[-–]\s*\d+|\+|to\s*\d+)?\s*years?\s*(?:of\s+)?"
    r"(?:relevant\s+|working\s+|industry\s+|post\s+qualification\s+)?experience"
    r"|experience\s+of\s+(\d+)\s*(?:[-–]\s*\d+)?\s*\+?\s*years?"
    r"|experience\s*[:\-]\s*(\d+)"
    r")",
    re.IGNORECASE,
)

_SENIORITY_ALLOWED: Dict[str, Set[str]] = {
    "intern_junior": {"intern_junior", "mid"},
    "mid":           {"mid", "senior"},
    "senior":        {"mid", "senior"},
}

_GENERIC_DESC_RE = re.compile(
    r"please refer (the )?(vacancy|advert|advertisement)", re.IGNORECASE
)

SCRAPER_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DESCRIPTION_SELECTORS: List[Dict[str, str]] = [
    {"class": "job-description"}, {"class": "vacancy-description"},
    {"class": "job-details"},     {"id": "job-description"},
    {"id": "vacancyDescription"}, {"class": "jd-content"},
    {"class": "description"},     {"class": "content-area"},
    {"class": "jobad-detail"},    {"class": "job-ad-content"},
]
SKILLS_SELECTORS: List[Dict[str, str]] = [
    {"class": "job-skills"},      {"class": "skills"},
    {"class": "required-skills"}, {"id": "skills"},
    {"class": "competencies"},
]
SKILL_TAG_CSS: List[str] = [
    "span.skill-tag", "span.badge", "li.skill",
    "div.skill-item", ".skills-list li", ".tag-list span",
]


# ──────────────────────────────────────────────────────────────
# CV seniority detection
# ──────────────────────────────────────────────────────────────
def estimate_cv_years(cv_text: str) -> int:
    years = [
        int(m.group(1))
        for m in _RE_YEARS.finditer(cv_text or "")
        if m.group(1).isdigit()
    ]
    return max(years) if years else 0


def estimate_cv_level(cv_text: str) -> str:
    if not cv_text:
        return "intern_junior"
    if _RE_INTERN.search(cv_text):
        return "intern_junior"
    if _RE_JUNIOR.search(cv_text):
        return "intern_junior"
    if _RE_SENIOR.search(cv_text):
        return "senior"
    yrs = estimate_cv_years(cv_text)
    if yrs >= 5:
        return "senior"
    if yrs >= 2:
        return "mid"
    return "intern_junior"


# ──────────────────────────────────────────────────────────────
# Job seniority detection & filtering
# ──────────────────────────────────────────────────────────────
def _job_min_years(text: str) -> Optional[int]:
    """Extract the minimum years of experience from text."""
    m = _RE_EXP_REQ.search(text or "")
    if not m:
        return None
    for g in m.groups():
        if g and str(g).isdigit():
            return int(g)
    return None


def classify_job_level(title: str, desc: str, raw_skills: str = "") -> str:
    """
    Classify a job as intern_junior / mid / senior.

    v6 changes:
      - Now accepts raw_skills as a third parameter so experience requirements
        embedded in the skills field are also considered.
      - Default changed from "intern_junior" to "mid" — most unlabelled IT
        engineering jobs are mid-level hires, not junior/intern.
      - Senior threshold lowered from >= 5 to >= 4 years so "minimum 4-5 years"
        correctly classifies as senior.
    """
    combined = (title or "") + " " + (desc or "") + " " + (raw_skills or "")
    if _RE_INTERN.search(combined) or _RE_JUNIOR.search(combined):
        return "intern_junior"
    yrs = _job_min_years(combined)
    if yrs is not None:
        if yrs >= 4:          # v6: was >= 5
            return "senior"
        if yrs >= 2:
            return "mid"
    if _RE_SENIOR.search(combined):
        return "senior"
    return "mid"              # v6: was "intern_junior"


def apply_seniority_handling(
    df: pd.DataFrame,
    cv_level: str,
    mode: str = "filter",
) -> pd.DataFrame:
    global _DATA_SKILL_MAP, _DATA_SKILL_MAP_BUILT

    df = df.copy()
    if "job_level" not in df.columns:
        df["job_level"] = df.apply(
            lambda r: classify_job_level(
                r.get("title", ""),
                r.get("desc_clean", "") or r.get("description", ""),
                r.get("raw_skills", ""),
            ),
            axis=1,
        )
    if mode == "loose":
        allowed = _SENIORITY_ALLOWED.get(cv_level, {"intern_junior", "mid"})
        return df[df["job_level"].isin(allowed)]
    return df[df["job_level"] == cv_level]


# ──────────────────────────────────────────────────────────────
# Safety helpers
# ──────────────────────────────────────────────────────────────
def is_probably_pdf(file_storage) -> bool:
    if not file_storage:
        return False
    try:
        pos = file_storage.stream.tell()
        file_storage.stream.seek(0)
        header = file_storage.stream.read(5)
        file_storage.stream.seek(pos)
        return header == b"%PDF-"
    except Exception:
        return False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────
# Preview helpers
# ──────────────────────────────────────────────────────────────
def cleanup_old_previews(ttl_seconds: int = PREVIEW_TTL_SECONDS) -> None:
    now = time.time()
    try:
        for name in os.listdir(PREVIEW_FOLDER):
            d = os.path.join(PREVIEW_FOLDER, name)
            if not os.path.isdir(d):
                continue
            try:
                if now - os.path.getmtime(d) > ttl_seconds:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def save_preview_file(src_pdf_path: str, original_filename: str) -> Dict[str, str]:
    preview_id = uuid.uuid4().hex
    safe_name  = secure_filename(original_filename) or "resume.pdf"
    dest_dir   = os.path.join(PREVIEW_FOLDER, preview_id)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(src_pdf_path, os.path.join(dest_dir, safe_name))
    return {"preview_id": preview_id, "preview_filename": safe_name}


# ──────────────────────────────────────────────────────────────
# OCR helper — preprocessing for higher accuracy
# ──────────────────────────────────────────────────────────────
def preprocess_for_ocr(img_input) -> np.ndarray:
    if isinstance(img_input, bytes):
        nparr = np.frombuffer(img_input, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(img_input, _PILImage.Image):
        img = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img = img_input
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")

    if img is None:
        raise ValueError("Could not decode image — cv2.imdecode returned None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )

    return thresh


def _easyocr_from_array(img_array: np.ndarray) -> str:
    if not _OCR_AVAILABLE or _easyocr_reader is None:
        return ""
    results = _easyocr_reader.readtext(img_array, detail=0, paragraph=True)
    return " ".join(results).strip()


def _easyocr_from_bytes(raw_bytes: bytes) -> str:
    try:
        arr = preprocess_for_ocr(raw_bytes)
        return _easyocr_from_array(arr)
    except Exception as e:
        logger.warning(f"EasyOCR (bytes) failed: {e}")
        return ""


# ──────────────────────────────────────────────────────────────
# PDF extraction bundle
# ──────────────────────────────────────────────────────────────
def extract_pdf_bundle(pdf_path: str) -> Dict[str, Any]:
    doc = None
    try:
        doc       = pymupdf.open(pdf_path)
        full_text: List[str] = []
        has_image = False
        links:    List[str] = []
        ocr_used  = False

        for page in doc:
            page_text = page.get_text().strip()

            if len(page_text) < 50 and _OCR_AVAILABLE:
                try:
                    mat = pymupdf.Matrix(300 / 72, 300 / 72)
                    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)

                    pil_img  = _PILImage.frombytes("L", (pix.width, pix.height), pix.samples)
                    bgr_arr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
                    arr      = preprocess_for_ocr(bgr_arr)
                    ocr_text = _easyocr_from_array(arr)

                    if len(ocr_text.strip()) > len(page_text):
                        page_text = ocr_text
                        ocr_used  = True
                        logger.info(
                            f"EasyOCR used on page {page.number + 1} "
                            f"({len(ocr_text)} chars extracted)"
                        )
                except Exception as ocr_err:
                    logger.warning(f"EasyOCR failed on page {page.number + 1}: {ocr_err}")

            full_text.append(page_text)

            if not has_image and page.get_images(full=True):
                has_image = True
            for link in page.get_links():
                uri = link.get("uri")
                if uri:
                    links.append(uri.strip())

        text = "\n".join(full_text)
        return {
            "text":       text,
            "text_lower": text.lower(),
            "has_image":  has_image,
            "ocr_used":   ocr_used,
            "links":      list(dict.fromkeys(links)),
            "page_count": len(doc),
            "markdown":   None,
        }
    finally:
        if doc:
            doc.close()


def get_markdown_cached(pdf_path: str, bundle: Dict[str, Any]) -> str:
    if bundle.get("markdown") is None:
        if bundle.get("ocr_used"):
            bundle["markdown"] = bundle.get("text", "")
        else:
            bundle["markdown"] = pymupdf4llm.to_markdown(pdf_path)
    return bundle["markdown"]


# ──────────────────────────────────────────────────────────────
# Evidence helper
# ──────────────────────────────────────────────────────────────
def find_evidence_snippets(
    text: str,
    patterns: List[str],
    max_hits: int = 3,
    window: int = 80,
) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start   = max(0, m.start() - window)
            end     = min(len(text), m.end() + window)
            snippet = text[start:end].replace("\n", " ").strip()
            out.append(snippet)
            if len(out) >= max_hits:
                return out
    return out


# ──────────────────────────────────────────────────────────────
# CV check functions
# ──────────────────────────────────────────────────────────────
def check_cv_page_count(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        page_count = bundle["page_count"] if bundle else len(pymupdf.open(pdf_path))
        if page_count == 1:
            return {"status": "success", "message": "Perfect! Single page CV.", "value": "1 page ✓", "score": 10}
        if page_count == 2:
            return {"status": "success", "message": "Good! Two pages is acceptable.", "value": "2 pages ✓", "score": 8}
        return {
            "status": "warning",
            "message": f"CV has {page_count} pages. Keep it to 1-2 pages.",
            "value": f"{page_count} pages",
            "score": 4,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_gpa_in_cv(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text  = bundle["text"]       if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        text_lower = bundle["text_lower"] if bundle else full_text.lower()

        gpa_patterns = [
            r"(?:current\s*)?gpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"cgpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"grade\s*point\s*(?:average)?\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"gpa\s*(?:is)?\s*(\d(?:\.\d{1,2})?)",
        ]

        found_value:   Optional[str] = None
        evidence_line: Optional[str] = None

        for line in (ln.strip() for ln in full_text.splitlines() if ln.strip()):
            for pat in gpa_patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    found_value   = m.group(1)
                    evidence_line = line
                    break
            if found_value:
                break

        if found_value:
            return {
                "status": "success",
                "message": "GPA is mentioned in the CV.",
                "value": f"GPA {found_value} ✓",
                "score": 10,
                "evidence": [evidence_line],
            }
        if "gpa" in text_lower or "cgpa" in text_lower:
            return {
                "status": "warning",
                "message": "GPA keyword found but value is unclear. Use format: 'Current GPA: 3.71'.",
                "value": "Unclear",
                "score": 6,
                "evidence": find_evidence_snippets(full_text, [r"\bgpa\b", r"\bcgpa\b"], max_hits=2, window=40),
            }
        return {
            "status": "warning",
            "message": "GPA not mentioned. If you have a good GPA (>3.0), add it.",
            "value": "Not found",
            "score": 5,
            "evidence": [],
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0, "evidence": []}


def check_professional_email(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text = bundle["text"] if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        emails    = list(dict.fromkeys(
            re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", full_text)
        ))

        if not emails:
            return {
                "status": "error",
                "message": "No email address found. Add a reachable email for recruiters.",
                "value": "Missing",
                "score": 0,
                "emails": [],
                "risk_level": "N/A",
                "suggestion": "Use a simple name-based email (e.g., firstname.lastname@gmail.com).",
            }

        local      = emails[0].split("@")[0].lower()
        slang      = {"cool", "hot", "sexy", "boss", "king", "queen", "swag", "ninja", "devil", "angel", "xoxo"}
        risk_points = 0
        reasons:   List[str] = []

        if any(w in local for w in slang):
            risk_points += 2
            reasons.append("contains informal word")
        if re.search(r"\d{4,}", local):
            risk_points += 1
            reasons.append("has long number sequence")
        if len(re.findall(r"[._-]", local)) > 3:
            risk_points += 1
            reasons.append("uses many symbols")

        if risk_points == 0:
            risk, score, msg = "Low",    10, "Email found. Looks fine for professional use."
        elif risk_points <= 2:
            risk, score, msg = "Medium",  8, "Email found. Consider a simpler name-based email."
        else:
            risk, score, msg = "High",    6, "Email found, but it may look informal."

        return {
            "status": "success",
            "message": msg,
            "value": "Email present ✓",
            "score": score,
            "emails": emails,
            "risk_level": risk,
            "reasons": reasons,
            "suggestion": "Preferred format: firstname.lastname@domain.com",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_photo_presence(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if bundle:
            has_image = bundle["has_image"]
        else:
            doc = pymupdf.open(pdf_path)
            try:
                has_image = any(doc[i].get_images(full=True) for i in range(len(doc)))
            finally:
                doc.close()

        if has_image:
            return {"status": "success", "message": "Photo found.", "value": "Present ✓", "score": 10}
        return {
            "status": "info",
            "message": "No photo detected. Consider adding a professional photo.",
            "value": "Not found",
            "score": 7,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_ol_al_presence(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text  = bundle["text"]       if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        text_lower = bundle["text_lower"] if bundle else full_text.lower()

        ol_patterns = ["o/l", "o.l", "ordinary level", "g.c.e o/l", "gce o/l"]
        al_patterns = ["a/l", "a.l", "advanced level", "g.c.e a/l", "gce a/l"]

        has_ol = any(p in text_lower for p in ol_patterns)
        has_al = any(p in text_lower for p in al_patterns)

        if not (has_ol or has_al):
            return {
                "status": "success",
                "message": "No school exam results found. Good — focus on degree and technical skills.",
                "value": "Not present ✓",
                "score": 10,
                "evidence": [],
            }

        regex_patterns = [
            r"\b(o\/l|o\.l|g\.c\.e\s*o\/l|gce\s*o\/l|ordinary level)\b",
            r"\b(a\/l|a\.l|g\.c\.e\s*a\/l|gce\s*a\/l|advanced level)\b",
        ]
        evidence_lines: List[str] = []
        for line in full_text.splitlines():
            lc = " ".join(line.split())
            if len(lc) < 6:
                continue
            if any(re.search(p, lc, flags=re.IGNORECASE) for p in regex_patterns):
                evidence_lines.append(lc)
            if len(evidence_lines) >= 3:
                break
        if not evidence_lines:
            evidence_lines = find_evidence_snippets(full_text, regex_patterns, max_hits=2, window=50)

        details = (["O/L"] if has_ol else []) + (["A/L"] if has_al else [])
        return {
            "status": "warning",
            "message": (
                f"School exam results ({', '.join(details)}) found. "
                "Sri Lankan IT companies typically don't require these."
            ),
            "value": "Present (consider removing)",
            "score": 5,
            "evidence": evidence_lines,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0, "evidence": []}


def check_formatting_quality(pdf_path: str) -> Dict[str, Any]:
    doc = None
    try:
        doc    = pymupdf.open(pdf_path)
        issues: List[str] = []
        font_sizes: Set[float] = set()

        for page_num in range(len(doc)):
            for block in doc[page_num].get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        try:
                            font_sizes.add(round(float(span.get("size", 0)), 1))
                        except Exception:
                            pass

        if len(font_sizes) < 2:
            issues.append("Use different font sizes for headings and body text")
        elif len(font_sizes) > 6:
            issues.append("Too many different font sizes — keep it consistent")

        if len(doc) > 0:
            blocks = doc[0].get_text("blocks")
            if blocks and min(b[0] for b in blocks) < 36:
                issues.append("Margins appear too small")

        score = max(0, min(10, 10 - len(issues) * 2))
        if issues:
            return {
                "status": "warning",
                "message": f"Formatting suggestions: {'; '.join(issues)}",
                "value": f"{score}/10",
                "score": score,
            }
        return {"status": "success", "message": "Formatting looks good.", "value": "Good ✓", "score": 10}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 5}
    finally:
        if doc:
            doc.close()


def validate_technical_keywords(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text = bundle["text"] if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        text      = re.sub(r"\s+", " ", full_text).lower()

        found: List[str] = []
        for label in SRI_LANKAN_TECH_KEYWORDS:
            patterns = TECH_KEYWORD_VARIANTS.get(label, [])
            for pat in patterns:
                if re.search(pat, text, flags=re.IGNORECASE):
                    found.append(label)
                    break

        found     = list(dict.fromkeys(found))
        kw_count  = len(found)

        if kw_count >= 10:
            status, msg, score = "success", f"Excellent! Found {kw_count} relevant technical keywords.", 10
        elif kw_count >= 5:
            status, msg, score = "success", f"Good! Found {kw_count} technical keywords. Consider adding more.", 7
        elif kw_count >= 3:
            status, msg, score = "warning", f"Basic technical keywords found ({kw_count}). Add more tools/frameworks.", 5
        else:
            status, msg, score = "warning", f"Only {kw_count} technical keywords found. Add a clear skills section.", 3

        return {
            "status": status,
            "message": msg,
            "value": f"{kw_count} keywords",
            "found_keywords": found[:12],
            "score": score,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


# ──────────────────────────────────────────────────────────────
# GitHub link functions
# ──────────────────────────────────────────────────────────────
def normalize_github_url(url: str) -> str:
    u = url.strip()
    if not u.lower().startswith("http"):
        u = "https://" + u
    return u.split("#")[0].split("?")[0].rstrip("/")


def is_github_repo_link(url: str) -> bool:
    parts = url.split("?")[0].rstrip("/").split("/")
    if len(parts) < 5:
        return False
    owner, repo = parts[3], parts[4]
    if not owner or not repo:
        return False
    non_repo_paths = {
        "followers", "following", "repositories", "repos",
        "stars", "starred", "packages", "projects", "settings",
    }
    return repo.lower() not in non_repo_paths


def github_url_exists(url: str) -> Tuple[bool, str]:
    norm = normalize_github_url(url)
    if norm in _GITHUB_URL_CACHE:
        return _GITHUB_URL_CACHE[norm]

    try:
        host = (urlparse(norm).hostname or "").lower()
        if host not in {"github.com", "www.github.com"}:
            result = (False, "Blocked (non-GitHub host)")
        else:
            r = requests.get(norm, timeout=HTTP_TIMEOUT, allow_redirects=True, headers=DEFAULT_UA)
            if r.status_code == 200:
                result = (True, "Working")
            elif r.status_code == 404:
                result = (False, "Not found")
            else:
                result = (False, f"HTTP {r.status_code}")
    except Exception:
        result = (False, "Connection error")

    _github_cache_set(norm, result)
    return result


def extract_github_links_from_text(text: str) -> List[str]:
    if not text:
        return []
    fixed   = re.sub(r"(github\.com/)\s+", r"\1", text, flags=re.IGNORECASE)
    fixed   = re.sub(r"(github\.com/[A-Za-z0-9_.-]+)\s+([A-Za-z0-9_.-]+)", r"\1\2", fixed, flags=re.IGNORECASE)
    pattern = r"(?:https?://)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?"
    out: List[str] = []
    for u in re.findall(pattern, fixed, flags=re.IGNORECASE):
        if not u.lower().startswith("http"):
            u = "https://" + u
        out.append(u)
    return list(dict.fromkeys(out))


def validate_github_links(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        links = [u for u in (bundle["links"] if bundle else []) if "github.com" in u.lower()]
        if not links:
            md    = get_markdown_cached(pdf_path, bundle) if bundle else pymupdf4llm.to_markdown(pdf_path)
            links = extract_github_links_from_text(md)

        links = list(dict.fromkeys(normalize_github_url(u) for u in links))
        if not links:
            return {"status": "warning", "message": "No GitHub links found.", "repos": [], "value": "No links", "score": 3}

        repo_links    = [u for u in links if is_github_repo_link(u)]
        profile_links = [u for u in links if u not in repo_links]
        details: List[Dict[str, Any]] = []
        valid_profile = valid_repo = 0

        for u in profile_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "profile", "valid": ok, "message": msg})
            if ok:
                valid_profile += 1

        for u in repo_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "repo", "valid": ok, "message": msg})
            if ok:
                valid_repo += 1

        if valid_profile == 0 and valid_repo == 0:
            status, score = "warning", 3
        else:
            base   = 6 if valid_profile > 0 else 0
            boost  = 4 if valid_repo >= 3 else 3 if valid_repo == 2 else 2 if valid_repo == 1 else 0
            score  = min(10, base + boost)
            status = "success" if score >= 7 else "warning"

        return {
            "status": status,
            "message": (
                f"GitHub found: {len(profile_links)} profile link(s), {len(repo_links)} repo link(s). "
                f"Working: {valid_profile} profile, {valid_repo} repo."
            ),
            "repos": details,
            "value": f"{valid_profile} profile, {valid_repo} repo working",
            "score": score,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "repos": [], "value": "Error", "score": 0}


def find_specialization(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    specializations = {
        "Software Technology":   "software technology",
        "Network Technology":    "network technology",
        "Multimedia Technology": "multimedia technology",
    }
    try:
        text = (bundle["text_lower"] if bundle else
                "\n".join(p.get_text().lower() for p in pymupdf.open(pdf_path)))[:12000]
        for spec_name, keyword in specializations.items():
            if keyword in text:
                return {"status": "success", "message": f"Specialization: {spec_name}", "value": spec_name, "score": 10}
        return {"status": "warning", "message": "Specialization area is not clearly mentioned.", "value": "Not specified", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


# ──────────────────────────────────────────────────────────────
# check_skills_separation — FULLY FIXED (v3)
# ──────────────────────────────────────────────────────────────
def check_skills_separation(pdf_path: str) -> Dict[str, Any]:
    doc = None
    try:
        doc        = pymupdf.open(pdf_path)
        all_blocks: List[Any] = []
        for page in doc:
            all_blocks.extend(page.get_text("blocks"))
        all_blocks.sort(key=lambda b: (b[1], b[0]))

        sorted_lines = [
            line.strip()
            for block in all_blocks
            for line in block[4].splitlines()
        ]

        soft_skills_raw:  List[str] = []
        technical_skills: List[str] = []
        current_section:  Optional[str] = None

        soft_headers = [
            "soft skill", "soft skills", "interpersonal",
            "personal skills", "personal qualities",
            "non-technical", "behavioural", "behavioral",
            "core skills", "key skills", "professional skills",
            "transferable skills", "strengths", "personal strengths",
        ]
        tech_headers = [
            "technical skill", "technical skills", "tech stack",
            "technologies", "programming", "tools", "frameworks",
            "technical expertise", "technical knowledge",
        ]
        stop_words = [
            "project", "experience", "education", "qualification",
            "reference", "contact", "certification", "certificate",
            "achievement", "profile", "summary", "declaration",
            "interest", "volunteer", "objective", "hobbies",
        ]

        for line in sorted_lines:
            if not line:
                continue
            ll       = line.lower()
            is_short = len(line) < 60

            if is_short and any(h in ll for h in soft_headers):
                current_section = "soft"
                continue
            if is_short and any(h in ll for h in tech_headers):
                current_section = "technical"
                continue
            if is_short and re.match(r"^skills\s*:?\s*$", ll):
                continue
            if is_short and any(w in ll for w in stop_words) and "skill" not in ll:
                current_section = None
                continue

            if current_section == "soft" and len(line) > 2:
                soft_skills_raw.append(line)
            elif current_section == "technical" and len(line) > 1:
                technical_skills.append(line)

        soft_skills_raw  = list(dict.fromkeys(soft_skills_raw))
        technical_skills = list(dict.fromkeys(technical_skills))

        extracted_soft_skills: List[str] = []
        for raw_line in soft_skills_raw:
            for canonical in _extract_from_line(raw_line):
                if canonical not in extracted_soft_skills:
                    extracted_soft_skills.append(canonical)

        base: Dict[str, Any] = {
            "extracted_soft_skills": extracted_soft_skills,
            "soft_skills_raw":       soft_skills_raw,
        }

        if soft_skills_raw and technical_skills:
            return {**base,
                "status": "success",
                "message": "Both Technical and Soft Skills sections found.",
                "value": "Both present ✓", "score": 10,
                "soft_count": len(soft_skills_raw),
                "tech_count": len(technical_skills),
            }
        if technical_skills:
            return {**base,
                "status": "warning",
                "message": "Technical skills found, but Soft Skills section is missing.",
                "value": "Partial", "score": 6,
                "tech_count": len(technical_skills), "soft_count": 0,
            }
        if soft_skills_raw:
            return {**base,
                "status": "warning",
                "message": "Soft Skills found, but Technical Skills section is missing.",
                "value": "Partial", "score": 5,
                "soft_count": len(soft_skills_raw), "tech_count": 0,
            }
        return {**base,
            "status": "error",
            "message": "Could not identify clear skill sections.",
            "value": "Not found", "score": 2,
            "soft_count": 0, "tech_count": 0,
        }

    except Exception as e:
        return {
            "status": "error", "message": f"Error: {e}",
            "value": "Error", "score": 0,
            "extracted_soft_skills": [], "soft_skills_raw": [],
        }
    finally:
        if doc:
            doc.close()


def check_contact_information(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text  = bundle["text"]       if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        text_lower = bundle["text_lower"] if bundle else full_text.lower()

        has_phone    = bool(re.search(r"(\+94|0)?[\s-]?[0-9]{9,10}", full_text))
        has_email    = bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", full_text))
        has_linkedin = "linkedin" in text_lower
        has_location = any(w in text_lower for w in ["colombo", "sri lanka", "address", "location"])

        score = 0
        found:   List[str] = []
        missing: List[str] = []

        for flag, pts, label in [
            (has_phone,    3, "Phone"),
            (has_email,    3, "Email"),
            (has_linkedin, 2, "LinkedIn"),
            (has_location, 2, "Location"),
        ]:
            if flag:
                score += pts
                found.append(label)
            else:
                missing.append(label)

        if score >= 8:
            return {"status": "success", "message": f"Complete contact info: {', '.join(found)}", "value": "Complete ✓", "score": 10, "details": found}
        if score >= 6:
            return {"status": "success", "message": f"Good contact info. Found: {', '.join(found)}", "value": "Good", "score": 8, "details": found}
        return {
            "status": "warning",
            "message": f"Missing: {', '.join(missing)}. Add complete contact information.",
            "value": "Incomplete",
            "score": max(4, score),
            "details": found,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_action_verbs(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    strong_verbs = [
        "developed", "designed", "implemented", "created", "built", "led", "managed",
        "optimized", "improved", "analyzed", "achieved", "delivered", "collaborated",
        "architected", "engineered", "deployed", "integrated", "automated", "streamlined",
    ]
    try:
        text_lower = bundle["text_lower"] if bundle else "\n".join(p.get_text().lower() for p in pymupdf.open(pdf_path))
        found_verbs = [v for v in strong_verbs if v in text_lower]
        n = len(found_verbs)

        if n >= 8:
            return {"status": "success", "message": f"Excellent! Found {n} strong action verbs.", "value": f"{n} verbs ✓", "score": 10, "verbs": found_verbs[:10]}
        if n >= 5:
            return {"status": "success", "message": f"Good use of action verbs ({n} found).", "value": f"{n} verbs", "score": 7, "verbs": found_verbs[:10]}
        if n >= 3:
            return {"status": "warning", "message": f"Only {n} action verbs found. Use more impact words.", "value": f"{n} verbs", "score": 5, "verbs": found_verbs[:10]}
        return {
            "status": "warning",
            "message": "Very few action verbs. Use words like 'developed', 'implemented', 'designed'.",
            "value": "Weak",
            "score": 3,
            "verbs": found_verbs[:10],
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_quantifiable_achievements(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        full_text = bundle["text"] if bundle else "\n".join(p.get_text() for p in pymupdf.open(pdf_path))
        patterns  = [
            r"\b\d+%\b",
            r"\b\d+\+\b",
            r"\b\d+\s*(projects?|users?|customers?|members?|students?)\b",
            r"\b(?:increased|decreased|improved|reduced|grew)\b[^\n]{0,60}\b\d+\b",
            r"\b\d+\s*(years?|months?)\b",
        ]
        hits = list(dict.fromkeys(
            m.group(0).strip()
            for pat in patterns
            for m in re.finditer(pat, full_text, flags=re.IGNORECASE)
            if m.group(0).strip()
        ))
        n = len(hits)
        if n >= 5:
            return {"status": "success", "message": f"Excellent! Found {n} quantifiable achievements.", "value": f"{n} metrics ✓", "score": 10, "examples": hits[:5]}
        if n >= 3:
            return {"status": "success", "message": f"Good quantification with {n} metrics.", "value": f"{n} metrics", "score": 7, "examples": hits[:5]}
        if n >= 1:
            return {"status": "warning", "message": f"Only {n} metrics found. Add more numbers to show impact.", "value": f"{n} metrics", "score": 5, "examples": hits[:5]}
        return {
            "status": "warning",
            "message": "No quantifiable achievements. Add numbers (e.g., '40% improvement', '5 projects').",
            "value": "None found",
            "score": 2,
            "examples": [],
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_professional_summary(pdf_path: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    doc = None
    try:
        doc        = pymupdf.open(pdf_path)
        first_page = doc[0]
        text       = first_page.get_text().lower()
        keywords   = ["summary", "profile", "objective", "career objective",
                      "professional summary", "about me", "introduction"]
        has_summary = any(k in text for k in keywords)
        blocks      = first_page.get_text("blocks")

        if blocks:
            top_section = first_page.rect.height * 0.3
            has_top = any(
                block[1] < top_section and any(k in (block[4] or "").lower() for k in keywords)
                for block in blocks
            )
            if has_top:
                return {"status": "success", "message": "Professional summary found at top of CV.", "value": "Present ✓", "score": 10}
            if has_summary:
                return {"status": "success", "message": "Summary section found.", "value": "Present", "score": 8}
            return {"status": "info", "message": "No professional summary. Consider adding a brief career objective.", "value": "Not found", "score": 6}

        return {"status": "warning", "message": "Could not detect professional summary.", "value": "Not detected", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc:
            doc.close()


# ──────────────────────────────────────────────────────────────
# Blind mode
# ──────────────────────────────────────────────────────────────
def apply_blind_mode(results: Dict[str, Any], blind_mode: bool) -> Dict[str, Any]:
    if not blind_mode:
        return results
    adjusted = dict(results)
    if "photo" in adjusted:
        adjusted["photo"] = {**adjusted["photo"], "message": "Blind mode: photo ignored for fairness.", "score": 10, "status": "info"}
    if "ol_al" in adjusted:
        adjusted["ol_al"] = {**adjusted["ol_al"], "message": "Blind mode: school results treated as low importance.", "score": max(adjusted["ol_al"].get("score", 0), 7)}
    return adjusted


# ──────────────────────────────────────────────────────────────
# Safe JSON parse
# ──────────────────────────────────────────────────────────────
def safe_json_loads(text: str) -> dict:
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ──────────────────────────────────────────────────────────────
# LLM validation
# ──────────────────────────────────────────────────────────────
def validate_with_llm(
    pdf_path: str,
    validation_results: Dict[str, Any],
    bundle: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    file_hash = sha256_file(pdf_path)
    if file_hash in _LLM_SCORE_CACHE:
        logger.info("LLM score served from cache (hash match).")
        return _LLM_SCORE_CACHE[file_hash]

    try:
        markdown_text = get_markdown_cached(pdf_path, bundle)
        groq_api_key  = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            return {"status": "error", "message": "API key missing.", "results": [], "passed": 0, "total": 9, "score": 0}

        prompt = f"""
Return STRICT JSON ONLY (no text, no markdown).
Output format:
{{
  "answers": [
    {{"id":1,"name":"Contact Info","yes":true,"evidence":"","suggestion":""}},
    {{"id":2,"name":"Professional Email","yes":true,"evidence":"","suggestion":""}},
    {{"id":3,"name":"Page Count","yes":true,"evidence":"","suggestion":""}},
    {{"id":4,"name":"Skills Section","yes":true,"evidence":"","suggestion":""}},
    {{"id":5,"name":"Technical Keywords","yes":true,"evidence":"","suggestion":""}},
    {{"id":6,"name":"GitHub Link","yes":true,"evidence":"","suggestion":""}},
    {{"id":7,"name":"Action Verbs","yes":true,"evidence":"","suggestion":""}},
    {{"id":8,"name":"Quantified Achievements","yes":true,"evidence":"","suggestion":""}},
    {{"id":9,"name":"Professional Summary","yes":true,"evidence":"","suggestion":""}}
  ]
}}

Rules:
- If "yes" is true, provide a short evidence quote (max 12 words) from the CV text.
- If "yes" is false, provide a suggestion.

CV TEXT:
{markdown_text}
"""
        client   = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            seed=42,
        )

        raw     = (response.choices[0].message.content or "").strip()
        data    = safe_json_loads(raw)
        answers = data.get("answers", [])
        total   = 9
        passed  = sum(1 for a in answers[:total] if a.get("yes") is True)
        score   = round((passed / total) * 10, 1)

        result = {
            "status": "success",
            "message": "AI analysis complete.",
            "results": answers[:total],
            "passed": passed,
            "total": total,
            "score": score,
        }

        _llm_cache_set(file_hash, result)
        return result

    except Exception as e:
        return {"status": "error", "message": f"AI Analysis error: {e}", "results": [], "passed": 0, "total": 9, "score": 0}


# ──────────────────────────────────────────────────────────────
# Overall score
# ──────────────────────────────────────────────────────────────
def calculate_overall_score(results: Dict[str, Any], llm_std: float = 0.0) -> Dict[str, Any]:
    weights: Dict[str, float] = {
        "page_count": 0.05, "gpa": 0.05, "professional_email": 0.08,
        "photo": 0.04,      "ol_al": 0.05, "formatting": 0.06,
        "specialization": 0.04, "github": 0.08, "skills": 0.08,
        "keywords": 0.10,   "contact": 0.06, "action_verbs": 0.07,
        "achievements": 0.07, "summary": 0.05, "llm": 0.10,
    }
    if llm_std >= 2.0:
        weights["llm"] = 0.05

    total_score = total_weight = 0.0
    for key, w in weights.items():
        s = results.get("llm", {}).get("score", 0) if key == "llm" else results.get(key, {}).get("score", 0)
        total_score  += float(s or 0) * float(w)
        total_weight += float(w)

    final = (total_score / total_weight) if total_weight else 0.0

    if final >= 8.5:
        grade, status = "Excellent",          "success"
    elif final >= 7.0:
        grade, status = "Good",               "success"
    elif final >= 5.5:
        grade, status = "Fair",               "warning"
    else:
        grade, status = "Needs Improvement",  "warning"

    return {
        "status": status,
        "score": round(final, 1),
        "grade": grade,
        "message": f"Overall CV Score: {round(final, 1)}/10 — {grade}",
        "llm_weight_adjusted": llm_std >= 2.0,
    }


# ──────────────────────────────────────────────────────────────
# Dimension scoring
# ──────────────────────────────────────────────────────────────
def _get_score10(results: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = results.get(key, {})
        s = v.get("score", default) if isinstance(v, dict) else default
        return float(s) if s is not None else float(default)
    except Exception:
        return float(default)


def _weighted_avg_10(items: List[Tuple[float, float]]) -> float:
    total = wsum = 0.0
    for s, w in items:
        total += float(s) * float(w)
        wsum  += float(w)
    return (total / wsum) if wsum else 0.0


def calculate_dimension_scores(results: Dict[str, Any]) -> Dict[str, Any]:
    impact_10  = _weighted_avg_10([
        (_get_score10(results, "achievements"), 0.45),
        (_get_score10(results, "action_verbs"),  0.35),
        (_get_score10(results, "summary"),       0.20),
    ])
    brevity_10 = _weighted_avg_10([
        (_get_score10(results, "page_count"), 0.75),
        (_get_score10(results, "ol_al"),      0.25),
    ])
    style_10   = _weighted_avg_10([
        (_get_score10(results, "formatting"),         0.70),
        (_get_score10(results, "professional_email"), 0.20),
        (_get_score10(results, "photo"),              0.10),
    ])
    skills_10  = _weighted_avg_10([
        (_get_score10(results, "skills"),   0.40),
        (_get_score10(results, "keywords"), 0.40),
        (_get_score10(results, "github"),   0.20),
    ])

    def to100(x: float) -> int:
        return int(round(max(0.0, min(100.0, x * 10.0)), 0))

    return {
        "impact":  to100(impact_10),
        "brevity": to100(brevity_10),
        "style":   to100(style_10),
        "skills":  to100(skills_10),
    }


# ──────────────────────────────────────────────────────────────
# Greeting
# ──────────────────────────────────────────────────────────────
def get_time_greeting(now: Optional[datetime] = None) -> str:
    h = (now or datetime.now()).hour
    if   5  <= h < 12: return "Good morning."
    elif 12 <= h < 17: return "Good afternoon."
    elif 17 <= h < 22: return "Good evening."
    else:              return "Good night."


# ──────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────
def _safe_to_csv(df: pd.DataFrame, path: str) -> None:
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        os.close(fd)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_cache(cache_path: str = CACHE_CSV) -> Dict[str, Tuple[str, str]]:
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, dtype=str).fillna("")
        if not {"url", "description", "raw_skills"}.issubset(df.columns):
            return {}
        return {row["url"]: (row["description"], row["raw_skills"]) for _, row in df.iterrows()}
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return {}


def save_cache(cache: Dict[str, Tuple[str, str]], cache_path: str = CACHE_CSV) -> None:
    rows = [{"url": u, "description": d, "raw_skills": s} for u, (d, s) in cache.items()]
    if rows:
        _safe_to_csv(pd.DataFrame(rows), cache_path)


# ──────────────────────────────────────────────────────────────
# Scraper helpers
# ──────────────────────────────────────────────────────────────
def _get_text_from_selectors(soup: BeautifulSoup, selectors: List[Dict[str, str]]) -> str:
    for sel in selectors:
        el = soup.find(attrs=sel)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 40:
                return text
    return ""


def _get_skill_tags(soup: BeautifulSoup) -> str:
    for css_sel in SKILL_TAG_CSS:
        tags = soup.select(css_sel)
        if tags:
            skills = [t.get_text(strip=True) for t in tags if t.get_text(strip=True)]
            if skills:
                return ", ".join(skills)
    return ""


def _fallback_body_text(soup: BeautifulSoup) -> str:
    for tag in soup(["nav", "header", "footer", "script", "style", "aside", "form"]):
        tag.decompose()
    best = ""
    for block in soup.find_all(["div", "section", "article", "table", "td"]):
        txt = block.get_text(separator=" ", strip=True)
        if "refer to" in txt.lower() or "refer the advert" in txt.lower() or len(txt) < 120:
            continue
        if len(txt) > len(best):
            best = txt
    return best[:2500] if best else ""


# ──────────────────────────────────────────────────────────────
# scrape_job_page — v7 FIXED: smart image filtering, TopJobs flyer detection
# ──────────────────────────────────────────────────────────────

# Images that are definitely UI chrome — always skip
_SKIP_IMG_SRC_RE = re.compile(
    r"logo|icon|banner|header|footer|nav|button|arrow|social|avatar"
    r"|application\.png|apply\.png|submit\.png|click\.png"   # TopJobs UI buttons
    r"|\.gif$|1x1|spacer|pixel|tracker|sprite|thumb"
    r"|background|bg[\-_]|separator|divider|star\.png|rating",
    re.IGNORECASE,
)

# High-priority: definitely a job-content image — OCR these first
_CONTENT_IMG_SRC_RE = re.compile(
    r"JobAdvertismentServlet|CVServlet|vac[\-_]?ad|flyer|advert"
    r"|job[\-_]?img|vacancy|poster|jd[\-_]|job[\-_]ad|advertisement"
    r"|JobAdvert|jobad|job_ad|vacancyimage|vacancy_image",
    re.IGNORECASE,
)

# Minimum characters an OCR result must have to be considered real job content
# (125 chars is "Apply Now / Click here" level — reject it)
_MIN_OCR_CONTENT_CHARS = 200


def scrape_job_page(url: str, session_obj: requests.Session) -> Tuple[str, str]:
    """
    Scrape a job listing page and return (description_text, raw_skills_csv).

    v7 changes (fixes ../images/application.png false positive):
      - Added application.png / apply.png / submit.png to the SKIP list so
        TopJobs UI button images are never passed to OCR.
      - Added _MIN_OCR_CONTENT_CHARS = 200: OCR results shorter than this are
        discarded (they contain only button text / watermarks, not job content).
      - Priority images (JobAdvertismentServlet, CVServlet, etc.) are tried
        FIRST in a separate pass before falling back to generic image sweep.
        This avoids wasting time on decorative images when a flyer is present.
      - Non-priority image sweep is only run when description is still short
        (< 300 chars) after the priority pass AND HTML extraction.
    """
    if not url or not url.startswith("http"):
        return "", ""

    from urllib.parse import urljoin

    try:
        resp = session_obj.get(url, headers=SCRAPER_HEADERS, timeout=25, allow_redirects=True)
        if resp.status_code != 200:
            return "", ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) Try known HTML selectors first
        description = _get_text_from_selectors(soup, DESCRIPTION_SELECTORS)

        # 2) Try skill badge/tag spans
        skill_tags = _get_skill_tags(soup)

        # 3) Image OCR — two-pass approach
        if _OCR_AVAILABLE:
            best_ocr = ""

            all_imgs = soup.find_all("img")

            # ── Pass A: Priority images only (job flyer servlet URLs) ────────
            priority_imgs = [
                img for img in all_imgs
                if _CONTENT_IMG_SRC_RE.search(img.get("src") or "")
                and not _SKIP_IMG_SRC_RE.search(img.get("src") or "")
            ]

            for img_tag in priority_imgs:
                src = (img_tag.get("src") or "").strip()
                try:
                    img_url  = urljoin(url, src)
                    img_resp = session_obj.get(img_url, timeout=20, headers=SCRAPER_HEADERS)
                    if img_resp.status_code != 200 or len(img_resp.content) < 5000:
                        continue
                    arr      = preprocess_for_ocr(img_resp.content)
                    ocr_text = _easyocr_from_array(arr)
                    if len(ocr_text) >= _MIN_OCR_CONTENT_CHARS and len(ocr_text) > len(best_ocr):
                        best_ocr = ocr_text
                        logger.info(f"✓ EasyOCR [priority] ({len(ocr_text)} chars) from: {src[:60]}")
                    if len(best_ocr) > 400:
                        break
                except Exception as ocr_e:
                    logger.warning(f"EasyOCR priority image failed ({src[:40]}): {ocr_e}")

            # ── Pass B: Generic sweep — only when still no good content ──────
            # Skip this pass entirely if we already have rich content
            if not best_ocr and len(description.strip()) < 300:
                for img_tag in all_imgs:
                    src = (img_tag.get("src") or "").strip()
                    if not src:
                        continue
                    # Skip UI chrome and already-tried priority images
                    if _SKIP_IMG_SRC_RE.search(src):
                        continue
                    if _CONTENT_IMG_SRC_RE.search(src):
                        continue  # already tried in Pass A
                    try:
                        img_url  = urljoin(url, src)
                        img_resp = session_obj.get(img_url, timeout=20, headers=SCRAPER_HEADERS)
                        if img_resp.status_code != 200 or len(img_resp.content) < 10000:
                            # 10 KB minimum for generic sweep — avoids small decorative images
                            continue
                        arr      = preprocess_for_ocr(img_resp.content)
                        ocr_text = _easyocr_from_array(arr)
                        # Only accept if it looks like real job content (>= 200 chars)
                        if len(ocr_text) >= _MIN_OCR_CONTENT_CHARS and len(ocr_text) > len(best_ocr):
                            best_ocr = ocr_text
                            logger.info(f"✓ EasyOCR [generic] ({len(ocr_text)} chars) from: {src[:60]}")
                        if len(best_ocr) > 400:
                            break
                    except Exception as ocr_e:
                        logger.warning(f"EasyOCR generic image failed ({src[:40]}): {ocr_e}")

            # Merge meaningful OCR text with HTML description
            if best_ocr:
                if len(best_ocr) > len(description.strip()):
                    description = best_ocr + "\n" + description
                else:
                    description = description + "\n" + best_ocr

        # 4) If still short → try table text
        if not description or len(description.strip()) < 150:
            for table in soup.find_all("table"):
                txt = table.get_text(separator=" ", strip=True)
                if len(txt) > 250 and "Sign Up" not in txt[:100]:
                    description = txt
                    break

        # 5) Final fallback body scan
        if not description or len(description.strip()) < 150:
            description = _fallback_body_text(soup)

        combined_for_skills = (description or "")
        if skill_tags:
            combined_for_skills += " " + skill_tags

        found_skills = extract_skills_from_text(combined_for_skills)
        raw_skills   = ", ".join(found_skills)

        return description.strip(), raw_skills

    except Exception as e:
        logger.warning(f"scrape_job_page error for {url}: {e}")
        return "", ""


def enrich_csv_with_descriptions(
    csv_path: str = CSV_PATH,
    output_path: Optional[str] = None,
    cache_path: str = CACHE_CSV,
    force_rescrape: bool = False,
) -> pd.DataFrame:
    if output_path is None:
        base, ext    = os.path.splitext(csv_path)
        output_path  = base + "_enriched" + ext

    # ── v7: Read from enriched CSV if it exists so --enrich patches existing
    # data rather than re-scraping everything from the raw original CSV.
    # This means rows already enriched correctly are preserved; only rows
    # that _needs_enrich() flags get re-scraped.
    input_path = output_path if os.path.exists(output_path) else csv_path
    logger.info(f"Reading from: {input_path}")
    df = pd.read_csv(input_path, dtype=str).fillna("")

    # Make sure raw CSV columns that may be missing in the enriched file exist
    if output_path != input_path:
        raw_df = pd.read_csv(csv_path, dtype=str).fillna("")
        for col in raw_df.columns:
            if col not in df.columns:
                df[col] = ""

    for col in ("description", "raw_skills"):
        if col not in df.columns:
            df[col] = ""

    cache      = load_cache(cache_path)
    session_obj = requests.Session()

    groq_api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    client = (
        OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        if groq_api_key
        else None
    )

    # Bad OCR phrases that indicate the description came from a UI button image
    # rather than the actual job flyer (v7 fix)
    _BAD_OCR_PHRASES = [
        "apply now", "click here", "submit application",
        "apply online", "register now", "sign up",
    ]

    def _needs_enrich(row: pd.Series) -> bool:
        """
        v7: Detect rows whose description came from a UI button image (bad OCR),
        is too short, or is missing job-requirement signals (experience, years,
        required skills) that should have come from the job flyer image.
        """
        if force_rescrape:
            return True
        desc = str(row.get("description", "")).strip().lower()
        if not desc or "please refer" in desc or len(desc) < 40:
            return True
        # Discard descriptions that look like they came from button OCR
        if any(phrase in desc for phrase in _BAD_OCR_PHRASES) and len(desc) < 300:
            return True
        # Re-scrape descriptions that lack job-requirement signals.
        # Threshold 600 chars: most real job descriptions with flyer content
        # are 600+ chars. Short descriptions (even 266–571 chars like Levein)
        # that have no experience/year/minimum/required signal are missing the
        # flyer content and should be refreshed.
        _requirement_signals = [
            "year", "experience", "minimum", "required", "requirement",
            "qualification", "degree", "must have", "should have",
        ]
        if len(desc) < 600 and not any(kw in desc for kw in _requirement_signals):
            return True
        return False

    to_enrich_mask = df.apply(_needs_enrich, axis=1)
    to_enrich      = df[to_enrich_mask]
    logger.info(f"Jobs to enrich: {len(to_enrich)} / {len(df)}")

    for seq_num, (idx, row) in enumerate(to_enrich.iterrows()):
        url   = str(row.get("url",   "")).strip()
        title = str(row.get("title", "")).strip()

        # Always re-scrape even if in cache when _needs_enrich returned True
        # (cache may hold the old bad value)
        try:
            scraped_desc, scraped_skills = scrape_job_page(url, session_obj=session_obj)
            if scraped_desc:
                df.at[idx, "description"] = scraped_desc
                df.at[idx, "raw_skills"]  = scraped_skills or df.at[idx, "raw_skills"]
                cache[url] = (df.at[idx, "description"], df.at[idx, "raw_skills"])
                logger.info(f"✓ scraped [{seq_num+1}] {title[:50]} → {str(df.at[idx, 'raw_skills'])[:60]}")
                time.sleep(2.0)
                if (seq_num + 1) % 10 == 0:
                    save_cache(cache, cache_path)
                continue

            if client is None:
                inferred = infer_skills_from_title(title)
                skills   = ", ".join(inferred)
                desc     = f"Role: {title}. Required skills: {skills}"
                df.at[idx, "description"] = desc
                df.at[idx, "raw_skills"]  = skills
                cache[url] = (desc, skills)
                logger.info(f"✓ inferred [{seq_num+1}] {title[:50]} → {skills[:60]}")
                time.sleep(2.0)
                if (seq_num + 1) % 10 == 0:
                    save_cache(cache, cache_path)
                continue

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                messages=[{
                    "role": "user",
                    "content": (
                        f'You are a Sri Lankan IT job expert.\n'
                        f'Generate a realistic job description for: "{title}"\n'
                        f'Company is based in Sri Lanka.\n\n'
                        f'Return ONLY this JSON format:\n'
                        f'{{\n'
                        f'  "description": "2-3 sentence job description",\n'
                        f'  "skills": "comma separated list of 6-10 required technical skills"\n'
                        f'}}'
                    ),
                }],
            )
            raw    = response.choices[0].message.content or ""
            data   = safe_json_loads(raw)
            desc   = data.get("description", "")
            skills = data.get("skills", "")

            if not desc:
                inferred = infer_skills_from_title(title)
                skills   = ", ".join(inferred)
                desc     = f"Role: {title}. Required skills: {skills}"

            df.at[idx, "description"] = desc
            df.at[idx, "raw_skills"]  = skills
            cache[url] = (desc, skills)
            logger.info(f"✓ LLM [{seq_num+1}] {title[:50]} → {skills[:60]}")

        except Exception as e:
            logger.warning(f"Enrichment failed for {title}: {e}")
            inferred = infer_skills_from_title(title)
            skills   = ", ".join(inferred)
            desc     = f"Role: {title}. Required skills: {skills}"
            df.at[idx, "description"] = desc
            df.at[idx, "raw_skills"]  = skills

        time.sleep(2.0)

        if (seq_num + 1) % 10 == 0:
            save_cache(cache, cache_path)

    save_cache(cache, cache_path)
    _safe_to_csv(df, output_path)
    logger.info(f"Enrichment done → saved: {output_path}")
    return df


# ──────────────────────────────────────────────────────────────
# Build combined fields for TF-IDF
# ──────────────────────────────────────────────────────────────
def build_combined_fields(df: pd.DataFrame) -> pd.DataFrame:
    global _DATA_SKILL_MAP, _DATA_SKILL_MAP_BUILT
    df = df.copy()
    defaults = {
        "title": "Not specified", "company": "Company Name Withheld",
        "location": "Sri Lanka",  "description": "", "raw_skills": "",
        "closing_date": "",       "url": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    def clean_desc(d: str) -> str:
        d = d.strip()
        return "" if (not d or (_GENERIC_DESC_RE.search(d) and len(d) < 120)) else d

    df["desc_clean"] = df["description"].apply(clean_desc)

    if not _DATA_SKILL_MAP_BUILT:
        _DATA_SKILL_MAP       = build_data_skill_map(df)
        _DATA_SKILL_MAP_BUILT = True

    # Build job_skill_list directly from CSV raw_skills + description so that
    # all skills present in the CSV are available for matching.
    df["job_skill_list"] = df.apply(
        lambda r: _extract_skills_for_data_map(
            str(r.get("raw_skills", "") or ""),
            str(r.get("desc_clean", "") or r.get("description", "") or ""),
        ),
        axis=1,
    )

    # Add inferred skills from title and then clean/normalise.
    df["inferred_skills"] = df["title"].apply(infer_skills_from_title)
    df["job_skill_list"] = df.apply(
        lambda r: clean_job_skill_list(
            list(dict.fromkeys(
                (r["job_skill_list"] or []) + (r["inferred_skills"] or [])
            ))
        ),
        axis=1,
    )
    df["job_skill_text"] = df["job_skill_list"].apply(lambda xs: " ".join(xs or []))

    # v6: pass raw_skills to classify_job_level for better experience detection
    df["job_level"] = df.apply(
        lambda r: classify_job_level(
            r.get("title", ""),
            r.get("desc_clean", "") or r.get("description", ""),
            r.get("raw_skills", ""),
        ),
        axis=1,
    )

    df["job_text"] = (
        df["title"]       + " " +
        df["company"]     + " " +
        df["desc_clean"]  + " " +
        df["raw_skills"]  + " " +
        df["job_skill_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# ──────────────────────────────────────────────────────────────
# Load job data — cached at module level
# ──────────────────────────────────────────────────────────────
_JOB_DF_CACHE: Optional[pd.DataFrame] = None


def load_job_data(csv_path: str = CSV_PATH, auto_enrich: bool = True) -> Optional[pd.DataFrame]:
    global _JOB_DF_CACHE
    if _JOB_DF_CACHE is not None:
        return _JOB_DF_CACHE
    try:
        if auto_enrich:
            base, ext     = os.path.splitext(csv_path)
            enriched_path = base + "_enriched" + ext
            df = (
                pd.read_csv(enriched_path, dtype=str).fillna("")
                if os.path.exists(enriched_path)
                else enrich_csv_with_descriptions(csv_path=csv_path)
            )
        else:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        _JOB_DF_CACHE = build_combined_fields(df)
        return _JOB_DF_CACHE
    except Exception as e:
        logger.error(f"Error loading job data: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Date filter
# ──────────────────────────────────────────────────────────────
def _parse_topjobs_date_series(s: pd.Series) -> pd.Series:
    s   = s.astype(str).str.strip()
    dt1 = pd.to_datetime(s, errors="coerce", format="%a %b %d %Y")
    dt2 = pd.to_datetime(s, errors="coerce")
    return dt1.combine_first(dt2)


def filter_active_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "closing_date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["closing_dt"] = _parse_topjobs_date_series(out["closing_date"].astype(str))
    today = pd.Timestamp.today().normalize()
    mask  = out["closing_dt"].isna() | (out["closing_dt"] >= today)
    return out[mask]


# ──────────────────────────────────────────────────────────────
# User skills from CV
# ──────────────────────────────────────────────────────────────
def build_user_skills_from_cv(results: Dict[str, Any], cv_text: str) -> List[str]:
    skills: List[str] = []

    kw = (results or {}).get("keywords", {}).get("found_keywords", [])
    if isinstance(kw, list):
        skills.extend(normalize_token(x) for x in kw)

    skills.extend(extract_skills_from_text(cv_text or ""))

    try:
        gh_msg = (results or {}).get("github", {}).get("message", "") or ""
        skills.extend(extract_skills_from_text(gh_msg))
    except Exception:
        pass

    # Use the permissive job-skill cleaner so CV skills are
    # normalised in the same way as job-side skills, improving
    # matched/missing skill overlap in recommendations.
    return clean_job_skill_list(skills)


# ──────────────────────────────────────────────────────────────
# Soft skill full-text fallback
# ──────────────────────────────────────────────────────────────
def _extract_soft_skills_fulltext(cv_text: str) -> List[str]:
    FALLBACK_PATTERNS: Dict[str, List[str]] = {
        "communication":       [r"\bcommunication\s*skill", r"\binterpersonal\b"],
        "teamwork":            [r"\bteam[\s\-]?work", r"\bteam\s+player", r"\bcollaborat\w+"],
        "problem_solving":     [r"\bproblem[\s\-]?solv\w+", r"\btroubleshoot\w+"],
        "leadership":          [r"\bleadership", r"\bmentor(?:ed|ing)?\b"],
        "time_management":     [r"\btime[\s\-]?management", r"\bdeadline", r"\bprioritiz\w+"],
        "adaptability":        [r"\badaptab\w+", r"\bquick\s+learn\w+"],
        "creativity":          [r"\bcreat(?:ive|ivity)", r"\binnovat\w+"],
        "attention_to_detail": [r"\battention\s+to\s+detail", r"\bdetail[\s\-]?oriented"],
        "customer_focus":      [r"\bcustomer[\s\-]?(?:focus|service)"],
        "self_motivation":     [r"\bself[\s\-]?motivat\w+", r"\bproactiv\w+"],
        "analytical_thinking": [r"\banalytical\b", r"\bcritical\s+think\w+"],
    }
    found: List[str] = []
    for skill, patterns in FALLBACK_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, cv_text, flags=re.IGNORECASE):
                found.append(skill)
                break
    return found


# ──────────────────────────────────────────────────────────────
# Main recommender
# ──────────────────────────────────────────────────────────────
_LEVEL_HINT: Dict[str, str] = {
    "intern_junior": "intern junior trainee entry level graduate",
    "mid":           "mid level experienced engineer developer",
    "senior":        "senior lead architect principal manager",
}


def get_job_recommendations(
    results: Dict[str, Any],
    cv_text: str,
    location_filter: str = "All",
    top_n: int = 15,
    seniority_mode: str = "filter",
    auto_enrich: bool = True,
) -> List[Dict[str, Any]]:
    try:
        df = load_job_data(auto_enrich=auto_enrich)
        if df is None or df.empty:
            return []

        user_skills = build_user_skills_from_cv(results, cv_text=cv_text)
        if not user_skills:
            logger.warning("No user skills extracted from CV.")
            return []

        section_soft: List[str]   = (results.get("skills") or {}).get("extracted_soft_skills") or []
        fulltext_soft: List[str]  = _extract_soft_skills_fulltext(cv_text or "")
        cv_soft_skills: List[str] = sorted(set(section_soft) | set(fulltext_soft))
        logger.info(f"Soft skills for matching: {cv_soft_skills}")

        if location_filter and location_filter != "All":
            df = df[df["location"].str.contains(location_filter, case=False, na=False)]
            if df.empty:
                return []

        df = filter_active_jobs(df)
        if df.empty:
            return []

        cv_level = estimate_cv_level(cv_text)

        # Ensure job_level is classified (may already be set in build_combined_fields)
        if "job_level" not in df.columns:
            df["job_level"] = df.apply(
                lambda r: classify_job_level(
                    r.get("title", ""),
                    r.get("desc_clean", "") or r.get("description", ""),
                    r.get("raw_skills", ""),
                ),
                axis=1,
            )

        level_hint = _LEVEL_HINT.get(cv_level, "junior entry level")
        query_text = (level_hint + " " + " ".join(user_skills)).strip()
        job_texts  = df["job_text"].fillna("").astype(str).tolist()

        vec = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
        )
        mat  = vec.fit_transform(job_texts + [query_text])
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()

        df = df.reset_index(drop=True)
        df["tfidf_score"] = sims

        cv_set: Set[str] = set(user_skills)

        def _overlap_ratio(job_skills: Any) -> float:
            js = set(clean_job_skill_list(job_skills or []))
            return len(js & cv_set) / len(js) if js else 0.3

        df["overlap"] = df["job_skill_list"].apply(_overlap_ratio)

        def _seniority_score(job_level: str) -> float:
            jl      = (job_level or "").strip() or "mid"
            allowed = _SENIORITY_ALLOWED.get(cv_level, set())
            return 1.0 if jl == cv_level else 0.7 if jl in allowed else 0.0

        df["seniority_score"] = df["job_level"].apply(_seniority_score)

        df["soft_score"] = df.apply(
            lambda row: compute_soft_skill_match(cv_soft_skills, str(row.get("title", "")))["score"],
            axis=1,
        )

        df["final_score"] = (
            df["tfidf_score"]     * 0.50 +
            df["overlap"]         * 0.30 +
            df["soft_score"]      * 0.10 +
            df["seniority_score"] * 0.10
        )

        df = (
            df.sort_values("final_score", ascending=False)
              .drop_duplicates(subset=["title", "company"], keep="first")
              .head(int(top_n))
        )

        _LEVEL_TO_FRONTEND = {
            "intern_junior": "intern",
            "mid":           "mid",
            "senior":        "senior",
        }

        jobs: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            job_title  = str(row.get("title", ""))
            job_skills = clean_job_skill_list(row.get("job_skill_list") or [])
            job_set    = set(job_skills)
            soft_info  = compute_soft_skill_match(cv_soft_skills, job_title)
            raw_desc   = str(row.get("description", ""))
            desc_short = (raw_desc[:220] + "...") if len(raw_desc) > 220 else raw_desc
            match_pct  = round(min(100.0, max(0.0, float(row["final_score"]) * 100.0)), 1)

            job_level_raw   = str(row.get("job_level", "mid") or "mid")
            seniority_level = _LEVEL_TO_FRONTEND.get(job_level_raw, "mid")

            jobs.append({
                "title":                job_title,
                "company":              row.get("company", ""),
                "location":             row.get("location", ""),
                "description":          desc_short or "Please refer the vacancy.",
                "url":                  row.get("url", ""),
                "closing_date":         row.get("closing_date", ""),
                "match_percentage":     match_pct,
                "match_level": (
                    "Excellent" if match_pct >= 70
                    else "Good" if match_pct >= 50
                    else "Potential"
                ),
                "seniority_level":      seniority_level,
                "cv_level_used":        cv_level,
                "matched_skills":       sorted(job_set & cv_set)[:8],
                "missing_skills":       sorted(job_set - cv_set)[:8],
                "soft_skills_matched":  soft_info["matched"],
                "soft_skills_missing":  soft_info["missing"],
                "soft_skills_expected": soft_info["job_expected"],
                "soft_score_display":   soft_info["score_display"],
                "cv_soft_skills":       cv_soft_skills,
            })

        results["matching_skills_used"] = user_skills
        results["cv_soft_skills"]       = cv_soft_skills
        return jobs

    except Exception as e:
        logger.exception(f"Error in get_job_recommendations: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────
def fetch_stats(days: int = 30) -> Dict[str, Any]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn   = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM resume_runs WHERE created_at >= ?", (cutoff,))
        total = cur.fetchone()[0] or 0

        cur.execute("SELECT AVG(overall_score) FROM resume_runs WHERE created_at >= ?", (cutoff,))
        avg_overall = round(cur.fetchone()[0] or 0, 2)

        cur.execute(
            "SELECT grade, COUNT(*) FROM resume_runs WHERE created_at >= ? GROUP BY grade ORDER BY COUNT(*) DESC",
            (cutoff,),
        )
        grade_dist = [{"grade": g, "count": c} for g, c in cur.fetchall()]

        cur.execute(
            """
            SELECT filename, overall_score, grade, llm_score, created_at
            FROM resume_runs WHERE created_at >= ?
            ORDER BY id DESC LIMIT 20
            """,
            (cutoff,),
        )
        recent = [
            {"filename": f, "overall_score": s, "grade": gr, "llm_score": ls, "created_at": ca}
            for f, s, gr, ls, ca in cur.fetchall()
        ]

        return {
            "days":               days,
            "total_runs":         total,
            "avg_overall":        avg_overall,
            "grade_distribution": grade_dist,
            "recent_runs":        recent,
        }
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
# Shared analysis logic
# ──────────────────────────────────────────────────────────────
def _run_analysis(
    filepath: str,
    filename: str,
    blind_mode: bool,
    location_filter: str,
    top_n: int = 15,
    seniority_mode: str = "filter",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    preview_info = save_preview_file(filepath, filename)
    file_hash    = sha256_file(filepath)
    bundle       = extract_pdf_bundle(filepath)
    ocr_was_used = bundle.get("ocr_used", False)

    if ocr_was_used:
        logger.info("OCR was used for this CV — image-based PDF detected.")

    results: Dict[str, Any] = {
        "ocr_used":           ocr_was_used,
        "page_count":         check_cv_page_count(filepath, bundle),
        "gpa":                check_gpa_in_cv(filepath, bundle),
        "professional_email": check_professional_email(filepath, bundle),
        "photo":              check_photo_presence(filepath, bundle),
        "ol_al":              check_ol_al_presence(filepath, bundle),
        "formatting":         check_formatting_quality(filepath),
        "specialization":     find_specialization(filepath, bundle),
        "github":             validate_github_links(filepath, bundle),
        "skills":             check_skills_separation(filepath),
        "keywords":           validate_technical_keywords(filepath, bundle),
        "contact":            check_contact_information(filepath, bundle),
        "action_verbs":       check_action_verbs(filepath, bundle),
        "achievements":       check_quantifiable_achievements(filepath, bundle),
        "summary":            check_professional_summary(filepath, bundle),
    }

    results["llm"]        = validate_with_llm(filepath, results, bundle)
    results               = apply_blind_mode(results, blind_mode=blind_mode)
    results["overall"]    = calculate_overall_score(results)
    results["dimensions"] = calculate_dimension_scores(results)
    results["preview_id"]       = preview_info["preview_id"]
    results["preview_filename"] = preview_info["preview_filename"]
    results["preview_url"]      = url_for(
        "preview_file",
        preview_id=preview_info["preview_id"],
        filename=preview_info["preview_filename"],
    )

    recommendations = get_job_recommendations(
        results=results,
        cv_text=bundle.get("text", ""),
        location_filter=location_filter,
        top_n=top_n,
        seniority_mode=seniority_mode,
    )
    results["job_recommendations"] = recommendations

    try:
        save_resume_run(
            file_hash=file_hash,
            filename=filename,
            overall_score=results["overall"]["score"],
            grade=results["overall"]["grade"],
            llm_score=results["llm"].get("score", 0),
            llm_std=0.0,
        )
    except Exception as exc:
        logger.warning(f"Could not save run to DB: {exc}")

    return results, recommendations


def _save_uploaded_file(file_storage) -> Tuple[str, str]:
    filename  = secure_filename(file_storage.filename)
    temp_name = f"{int(time.time())}_{filename}"
    filepath  = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
    file_storage.save(filepath)
    return filepath, filename


def _remove_file(filepath: str) -> None:
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route("/stats")
def stats():
    return render_template("stats.html", stats=fetch_stats())


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return render_template("index.html", results=None)

    cleanup_old_previews()

    if "cv_file" not in request.files:
        return render_template("index.html", results=None, error="Please select a CV file."), 400
    file = request.files["cv_file"]
    if file.filename == "":
        return render_template("index.html", results=None, error="Please select a CV file."), 400
    if not is_probably_pdf(file):
        return render_template("index.html", results=None, error="Please upload a valid PDF file only."), 400

    blind_mode      = request.form.get("blind_mode", "off") == "on"
    location_filter = request.form.get("location_filter", "All")
    filepath, filename = _save_uploaded_file(file)

    try:
        results, recommendations = _run_analysis(
            filepath, filename, blind_mode, location_filter, top_n=15
        )

        rec_key = uuid.uuid4().hex
        _rec_cache_set(rec_key, recommendations)
        session["rec_key"]       = rec_key
        session["last_filename"] = filename

        return render_template(
            "index.html",
            results=results,
            filename=filename,
            greeting=get_time_greeting(),
            sub_greeting="Welcome to your resume review.",
            job_recommendations=recommendations,
        )
    finally:
        gc.collect()
        _remove_file(filepath)


@app.route("/api/validate", methods=["POST"])
def api_validate():
    cleanup_old_previews()

    if "cv_file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["cv_file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not is_probably_pdf(file):
        return jsonify({"error": "Only valid PDF files accepted"}), 400

    blind_mode      = request.form.get("blind_mode", "off") == "on"
    location_filter = request.form.get("location_filter", "All")
    top_n           = int(request.form.get("top_n", 15))
    seniority_mode  = request.form.get("seniority_mode", "filter")
    filepath, filename = _save_uploaded_file(file)

    try:
        results, _ = _run_analysis(filepath, filename, blind_mode, location_filter, top_n, seniority_mode)
        return jsonify(results)
    finally:
        _remove_file(filepath)


@app.route("/job-recommendations", endpoint="job_recommendations")
def job_recommendations_page():
    rec_key  = session.get("rec_key", "")
    jobs     = _REC_CACHE.get(rec_key, [])
    filename = session.get("last_filename", "")
    return render_template(
        "job_recommendation.html",
        results={"job_recommendations": jobs},
        filename=filename,
    )


@app.route("/preview/<preview_id>/<filename>")
def preview_file(preview_id: str, filename: str):
    preview_id = re.sub(r"[^a-f0-9]", "", preview_id.lower())
    filename   = secure_filename(filename)
    user_dir   = os.path.join(PREVIEW_FOLDER, preview_id)
    full_path  = os.path.join(user_dir, filename)
    if not preview_id or not filename or not os.path.exists(full_path):
        abort(404)
    return send_from_directory(user_dir, filename, mimetype="application/pdf")


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Validator App / TopJobs enricher")
    parser.add_argument("--enrich",       action="store_true", help="Re-scrape only rows that need it (reads enriched CSV if exists)")
    parser.add_argument("--patch",        action="store_true", help="Alias for --enrich (patch bad/short rows in enriched CSV)")
    parser.add_argument("--csv",          default="topjobs_it_jobs.csv",           help="Input CSV path")
    parser.add_argument("--cache",        default="topjobs_description_cache.csv", help="Cache CSV path")
    parser.add_argument("--force",        action="store_true",                     help="Force re-scrape ALL rows (slow — full rebuild)")
    parser.add_argument("--reset-cache",  action="store_true",                     help="Delete enriched CSV + cache then rebuild from scratch (very slow)")

    args, _ = parser.parse_known_args()

    if args.reset_cache or args.enrich or args.patch:
        print("\n" + "=" * 55)
        if args.reset_cache:
            print("  MODE: Full reset — delete & rebuild from scratch")
        elif args.force:
            print("  MODE: Force re-scrape ALL rows")
        else:
            print("  MODE: Patch — re-scrape only bad/short rows")
        print("=" * 55)

        if args.reset_cache:
            base, ext     = os.path.splitext(args.csv)
            enriched_path = base + "_enriched" + ext
            for p in [enriched_path, args.cache]:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  Deleted: {p}")
            print()

        enrich_csv_with_descriptions(
            csv_path=args.csv,
            cache_path=args.cache,
            force_rescrape=args.force,
        )
        print("\n--- ENRICHMENT PROCESS COMPLETE ---")
    else:
        print("\n" + "=" * 55)
        print("  Enhanced CV Validator — Sri Lankan IT Market (v6)")
        print("  Access at: http://127.0.0.1:5000")
        print("=" * 55 + "\n")

        import platform
        is_windows = platform.system() == "Windows"

        # Python 3.14 on Windows raises WinError 10038 ("not a socket") when
        # Werkzeug's stat-reloader uses select() on a non-socket fd.
        # Fix: disable use_reloader on Windows so the broken select() path is
        # never entered.  Debug mode stays on so error pages still work.
        app.run(
            debug=True,
            host="0.0.0.0",
            port=5000,
            use_reloader=not is_windows,   # reloader off on Windows
        )