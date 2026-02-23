from flask import Flask, request, render_template, jsonify, abort, send_from_directory, url_for, session
import os
import re
import time
import gc
import hashlib
import sqlite3
import random
import logging
import argparse
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Set, Optional, Tuple
import json
import uuid
import shutil

import requests
import pymupdf
import pymupdf4llm
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = None

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

CSV_PATH = "topjobs_it_jobs.csv"
CACHE_CSV = "topjobs_description_cache.csv"

REQUEST_DELAY = (1.2, 2.6)
REQUEST_TIMEOUT = 12
MAX_RETRIES = 2

# ──────────────────────────────────────────────────────────────
# SQLite
# ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "data", "scores.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT,
                filename TEXT,
                overall_score REAL,
                grade TEXT,
                llm_score REAL,
                llm_std REAL,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_resume_run(
    file_hash: str, filename: str, overall_score: float,
    grade: str, llm_score: float, llm_std: float,
):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resume_runs
                (file_hash, filename, overall_score, grade, llm_score, llm_std, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_hash, filename, overall_score, grade, llm_score, llm_std,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


init_db()

# ──────────────────────────────────────────────────────────────
# TECH KEYWORD VARIANTS  (single definition — used everywhere)
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
    "Azure":            [r"\bazure\b"],
    "GCP":              [r"\bgcp\b", r"\bgoogle\s+cloud\b"],
    "Docker":           [r"\bdocker\b"],
    "Kubernetes":       [r"\bkubernetes\b", r"\bk8s\b"],
    "REST API":         [r"\brest(?:ful)?\s*api(?:s)?\b"],
    "GraphQL":          [r"\bgraphql\b"],
    "Microservices":    [r"\bmicroservices?\b"],
    "Agile":            [r"\bagile\b"],
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
}

# Precomputed set of normalized tech skill names for fast lookup
KNOWN_TECH_SKILLS: Set[str] = set()

SRI_LANKAN_TECH_KEYWORDS = [
    "Java", "Python", "JavaScript", "C#", "C++", "MySQL", "PostgreSQL", "MongoDB",
    "Oracle", "Git", "GitHub", "GitLab", "HTML5", "CSS3", "React", "Angular",
    "Vue.js", "Node.js", "Express", "Django", "Flask", "Spring Boot", "Android",
    "iOS", "Flutter", "React Native", "AWS", "Azure", "Docker", "Kubernetes",
    "REST API", "GraphQL", "Microservices", "Agile", "Scrum", "JIRA",
]

# ──────────────────────────────────────────────────────────────
# TITLE → SKILL INFERENCE MAP  (single definition)
# ──────────────────────────────────────────────────────────────
TITLE_SKILL_MAP: Dict[str, List[str]] = {
    "react":             ["React", "JavaScript", "HTML5", "CSS3", "Node.js", "Redux"],
    "angular":           ["Angular", "TypeScript", "JavaScript", "HTML5", "CSS3"],
    "vue":               ["Vue.js", "JavaScript", "HTML5", "CSS3"],
    "frontend":          ["JavaScript", "HTML5", "CSS3", "React"],
    "node":              ["Node.js", "JavaScript", "Express", "MongoDB", "REST API"],
    "python":            ["Python", "Django", "Flask", "SQL", "REST API"],
    "django":            ["Python", "Django", "SQL", "REST API"],
    "flask":             ["Python", "Flask", "REST API", "SQL"],
    "fastapi":           ["Python", "FastAPI", "REST API", "PostgreSQL"],
    "java":              ["Java", "Spring Boot", "SQL", "REST API"],
    "spring":            ["Java", "Spring Boot", "SQL", "REST API"],
    "php":               ["PHP", "MySQL", "HTML5", "CSS3", "Laravel"],
    "laravel":           ["PHP", "Laravel", "MySQL", "REST API"],
    "dotnet":            ["C#", "ASP.NET", "SQL", "REST API"],
    ".net":              ["C#", "ASP.NET", "SQL", "REST API"],
    "fullstack":         ["JavaScript", "React", "Node.js", "SQL", "REST API"],
    "full stack":        ["JavaScript", "React", "Node.js", "SQL", "REST API"],
    "backend":           ["REST API", "SQL", "Python", "Node.js"],
    "devops":            ["Docker", "Kubernetes", "CI/CD", "Linux", "AWS"],
    "cloud":             ["AWS", "Azure", "GCP", "Docker", "Terraform"],
    "aws":               ["AWS", "Docker", "Terraform", "Linux", "CI/CD"],
    "data engineer":     ["Python", "SQL", "Spark", "Hadoop"],
    "data scientist":    ["Python", "Machine Learning", "Pandas", "NumPy", "scikit-learn"],
    "machine learning":  ["Python", "TensorFlow", "PyTorch", "scikit-learn", "Pandas"],
    "ml engineer":       ["Python", "TensorFlow", "PyTorch", "Docker", "REST API"],
    "android":           ["Android", "Kotlin", "Java", "REST API"],
    "ios":               ["iOS", "Swift", "REST API"],
    "mobile":            ["Android", "iOS", "Flutter", "React Native"],
    "flutter":           ["Flutter", "Android", "iOS", "REST API"],
    "qa":                ["Selenium", "JIRA", "Agile", "Python"],
    "quality assurance": ["Selenium", "JIRA", "Agile"],
    "automation":        ["Selenium", "Python", "CI/CD"],
    "database":          ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle"],
    "ui":                ["Figma", "HTML5", "CSS3", "JavaScript", "React"],
    "ux":                ["Figma", "HTML5", "CSS3", "JavaScript"],
    "security":          ["Linux", "Python", "AWS"],
    "network":           ["Linux"],
}

# ──────────────────────────────────────────────────────────────
# SKILL NORMALIZATION  (single definition)
# ──────────────────────────────────────────────────────────────
_SKILL_SYNONYMS: Dict[str, str] = {
    "js":           "javascript",
    "reactjs":      "react",
    "node":         "node.js",
    "nodejs":       "node.js",
    "expressjs":    "express",
    "postgres":     "postgresql",
    "mongo":        "mongodb",
    "k8s":          "kubernetes",
    "ml":           "machine learning",
    "dl":           "deep learning",
    "tf":           "tensorflow",
    "springboot":   "spring boot",
    "restapi":      "rest api",
    "rest":         "rest api",
}

# Words that look like skills but are seniority/employment-type words
_GENERIC_EMPLOYMENT_WORDS: Set[str] = {
    "intern", "internship", "trainee", "entry level", "entry-level",
    "junior", "associate", "senior", "lead", "mid", "mid-level",
    "graduate", "undergraduate", "student", "fresher", "probationary",
    "full time", "full-time", "part time", "part-time",
    "contract", "temporary", "permanent",
}


def normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return _SKILL_SYNONYMS.get(t, t)


def normalize_keywords(items: List[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for k in items or []:
        nk = normalize_token(k)
        if nk:
            seen[nk] = None
    return list(seen)


def _build_known_tech_skills() -> Set[str]:
    """Build canonical set of normalized tech skill names from TECH_KEYWORD_VARIANTS keys."""
    s: Set[str] = set()
    for label in TECH_KEYWORD_VARIANTS:
        s.add(normalize_token(label))
    return s


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract recognized tech skills from free text using TECH_KEYWORD_VARIANTS patterns.
    Returns normalized skill names.
    """
    if not text:
        return []
    t = " " + re.sub(r"\s+", " ", text) + " "
    found: List[str] = []
    for label, patterns in TECH_KEYWORD_VARIANTS.items():
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                found.append(normalize_token(label))
                break
    return list(dict.fromkeys(found))


# Keep backward-compatible alias used throughout the file
extract_job_skills = extract_skills_from_text


def infer_skills_from_title(title: str) -> List[str]:
    title_lower = (title or "").lower()
    inferred: List[str] = []
    for keyword, skills in TITLE_SKILL_MAP.items():
        if keyword in title_lower:
            inferred.extend(skills)
    return list(dict.fromkeys(normalize_token(s) for s in inferred if s))


def clean_skill_list(skills: Any) -> List[str]:
    """
    Accept a list OR comma-separated string of skills.
    Keep only recognized tech skills; drop employment-type words.
    """
    if not skills:
        return []
    if isinstance(skills, str):
        parts = re.split(r"[,|/•\n]+", skills)
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


# Build the set once after TECH_KEYWORD_VARIANTS is defined
KNOWN_TECH_SKILLS = _build_known_tech_skills()

# ──────────────────────────────────────────────────────────────
# SENIORITY — single, authoritative set of regex patterns
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
_RE_YEARS = re.compile(r"(\d+)\+?\s*years?\b", re.IGNORECASE)
_RE_EXP_REQ = re.compile(
    r"(?:"
    r"(?:minimum|at\s+least|more\s+than|over"
    r"|should\s+have|must\s+have|require[sd]?)?\s*"
    r"(\d+)\s*(?:\+|to\s*\d+)?\s*years?\s*(?:of\s+)?"
    r"(?:relevant\s+|working\s+|industry\s+)?experience"
    r"|experience\s+of\s+(\d+)\s*\+?\s*years?"
    r"|experience\s*[:\-]\s*(\d+)"
    r")",
    re.IGNORECASE,
)

# Allowed job levels per CV level (loose mode)
_SENIORITY_ALLOWED: Dict[str, Set[str]] = {
    "intern_junior": {"intern_junior", "mid"},
    "mid":           {"intern_junior", "mid", "senior"},
    "senior":        {"mid", "senior"},
}

_GENERIC_DESC_RE = re.compile(
    r"please refer (the )?(vacancy|advert|advertisement)", re.IGNORECASE
)

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DESCRIPTION_SELECTORS = [
    {"class": "job-description"}, {"class": "vacancy-description"},
    {"class": "job-details"}, {"id": "job-description"},
    {"id": "vacancyDescription"}, {"class": "jd-content"},
    {"class": "description"}, {"class": "content-area"},
    {"class": "jobad-detail"}, {"class": "job-ad-content"},
]
SKILLS_SELECTORS = [
    {"class": "job-skills"}, {"class": "skills"},
    {"class": "required-skills"}, {"id": "skills"},
    {"class": "competencies"},
]
SKILL_TAG_CSS = [
    "span.skill-tag", "span.badge", "li.skill",
    "div.skill-item", ".skills-list li", ".tag-list span",
]


# ──────────────────────────────────────────────────────────────
# CV SENIORITY DETECTION
# ──────────────────────────────────────────────────────────────
def estimate_cv_years(cv_text: str) -> int:
    years = [
        int(m.group(1)) for m in _RE_YEARS.finditer(cv_text or "")
        if m.group(1).isdigit()
    ]
    return max(years) if years else 0


def estimate_cv_level(cv_text: str) -> str:
    """
    Returns one of: 'intern_junior', 'mid', 'senior'.
    Order matters: intern check before senior to avoid "senior intern" edge-cases.
    """
    if not cv_text:
        return "intern_junior"
    t = cv_text  # keep original case for regex (all patterns use IGNORECASE)

    # 1. Explicit intern keyword wins immediately
    if _RE_INTERN.search(t):
        return "intern_junior"

    # 2. Junior/trainee/graduate hints
    if _RE_JUNIOR.search(t):
        return "intern_junior"

    # 3. Explicit senior title
    if _RE_SENIOR.search(t):
        return "senior"

    # 4. Year-based heuristic
    yrs = estimate_cv_years(t)
    if yrs >= 5:
        return "senior"
    if yrs >= 2:
        return "mid"

    return "intern_junior"


# ──────────────────────────────────────────────────────────────
# JOB SENIORITY DETECTION & FILTERING
# ──────────────────────────────────────────────────────────────
def _job_min_years(text: str) -> Optional[int]:
    m = _RE_EXP_REQ.search(text or "")
    if not m:
        return None
    for g in m.groups():
        if g and str(g).isdigit():
            return int(g)
    return None


def classify_job_level(title: str, desc: str) -> str:
    combined = ((title or "") + " " + (desc or ""))

    if _RE_INTERN.search(combined) or _RE_JUNIOR.search(combined):
        return "intern_junior"

    yrs = _job_min_years(combined)
    if yrs is not None:
        if yrs >= 5:
            return "senior"
        if yrs >= 2:
            return "mid"

    if _RE_SENIOR.search(combined):
        return "senior"

    return "intern_junior"


def apply_seniority_handling(
    df: pd.DataFrame, cv_level: str, mode: str = "filter"
) -> pd.DataFrame:
    df = df.copy()

    if "job_level" not in df.columns:
        df["job_level"] = df.apply(
            lambda r: classify_job_level(
                r.get("title", ""),
                r.get("desc_clean", "") or r.get("description", ""),
            ),
            axis=1,
        )

    if mode == "loose":
        allowed = _SENIORITY_ALLOWED.get(cv_level, {"intern_junior", "mid"})
        return df[df["job_level"].isin(allowed)]

    # strict
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
def cleanup_old_previews(ttl_seconds: int = PREVIEW_TTL_SECONDS):
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
# PDF extraction bundle
# ──────────────────────────────────────────────────────────────
def extract_pdf_bundle(pdf_path: str) -> Dict[str, Any]:
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = []
        has_image = False
        links: List[str] = []

        for page in doc:
            full_text.append(page.get_text())
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
            "links":      list(dict.fromkeys(links)),
            "page_count": len(doc),
            "markdown":   None,
        }
    finally:
        if doc:
            doc.close()


def get_markdown_cached(pdf_path: str, bundle: Dict[str, Any]) -> str:
    if bundle.get("markdown") is None:
        bundle["markdown"] = pymupdf4llm.to_markdown(pdf_path)
    return bundle["markdown"]


# ──────────────────────────────────────────────────────────────
# Evidence helper
# ──────────────────────────────────────────────────────────────
def find_evidence_snippets(
    text: str, patterns: List[str], max_hits: int = 3, window: int = 80
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
# CV Check functions
# ──────────────────────────────────────────────────────────────
def check_cv_page_count(pdf_path: str, bundle: Dict[str, Any] = None):
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


def check_gpa_in_cv(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text  = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text_lower = full_text.lower()

        gpa_patterns = [
            r"(?:current\s*)?gpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"cgpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"grade\s*point\s*(?:average)?\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"gpa\s*(?:is)?\s*(\d(?:\.\d{1,2})?)",
        ]

        found_value = evidence_line = None
        for line in [ln.strip() for ln in full_text.splitlines() if ln.strip()]:
            for pat in gpa_patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    found_value   = m.group(1)
                    evidence_line = line
                    break
            if found_value:
                break

        if found_value:
            return {"status": "success", "message": "GPA is mentioned in the CV.",
                    "value": f"GPA {found_value} ✓", "score": 10, "evidence": [evidence_line]}
        if "gpa" in text_lower or "cgpa" in text_lower:
            return {
                "status": "warning",
                "message": "GPA keyword found, but GPA value is not clearly written. Use format like 'Current GPA: 3.71'.",
                "value": "Unclear", "score": 6,
                "evidence": find_evidence_snippets(full_text, [r"\bgpa\b", r"\bcgpa\b"], max_hits=2, window=40),
            }
        return {"status": "warning", "message": "GPA not mentioned. If you have a good GPA (>3.0), add it.",
                "value": "Not found", "score": 5, "evidence": []}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0, "evidence": []}


def check_professional_email(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        emails = list(dict.fromkeys(
            re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", full_text)
        ))

        if not emails:
            return {
                "status": "error", "message": "No email address found. Add a reachable email for recruiters.",
                "value": "Missing", "score": 0, "emails": [], "risk_level": "N/A",
                "suggestion": "Use a simple name-based email (e.g., firstname.lastname@gmail.com).",
            }

        local = emails[0].split("@")[0].lower()
        slang = {"cool","hot","sexy","boss","king","queen","swag","ninja","devil","angel","xoxo"}
        risk_points = 0
        reasons: List[str] = []

        if any(w in local for w in slang):
            risk_points += 2; reasons.append("contains informal word")
        if re.search(r"\d{4,}", local):
            risk_points += 1; reasons.append("has long number sequence")
        if len(re.findall(r"[._-]", local)) > 3:
            risk_points += 1; reasons.append("uses many symbols")

        if risk_points == 0:
            risk, score, msg = "Low",    10, "Email found. Looks fine for professional use."
        elif risk_points <= 2:
            risk, score, msg = "Medium",  8, "Email found. Consider a simpler name-based email."
        else:
            risk, score, msg = "High",    6, "Email found, but it may look informal."

        return {
            "status": "success", "message": msg, "value": "Email present ✓",
            "score": score, "emails": emails, "risk_level": risk, "reasons": reasons,
            "suggestion": "Preferred format: firstname.lastname@domain.com",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_photo_presence(pdf_path: str, bundle: Dict[str, Any] = None):
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
        return {"status": "info", "message": "No photo detected. Consider adding a professional photo.",
                "value": "Not found", "score": 7}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_ol_al_presence(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text  = (bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])) or ""
        text_lower = (bundle["text_lower"] if bundle else full_text.lower())

        ol_patterns = ["o/l", "o.l", "ordinary level", "g.c.e o/l", "gce o/l"]
        al_patterns = ["a/l", "a.l", "advanced level", "g.c.e a/l", "gce a/l"]

        has_ol = any(p in text_lower for p in ol_patterns)
        has_al = any(p in text_lower for p in al_patterns)

        if not (has_ol or has_al):
            return {"status": "success",
                    "message": "No school exam results found. Good - focus on degree and technical skills.",
                    "value": "Not present ✓", "score": 10, "evidence": []}

        patterns = [
            r"\b(o\/l|o\.l|g\.c\.e\s*o\/l|gce\s*o\/l|ordinary level)\b",
            r"\b(a\/l|a\.l|g\.c\.e\s*a\/l|gce\s*a\/l|advanced level)\b",
        ]
        evidence_lines: List[str] = []
        for line in full_text.splitlines():
            lc = " ".join(line.split())
            if len(lc) < 6:
                continue
            if any(re.search(p, lc, flags=re.IGNORECASE) for p in patterns):
                evidence_lines.append(lc)
            if len(evidence_lines) >= 3:
                break
        if not evidence_lines:
            evidence_lines = find_evidence_snippets(full_text, patterns, max_hits=2, window=50)

        details = (["O/L"] if has_ol else []) + (["A/L"] if has_al else [])
        return {
            "status": "warning",
            "message": f"School exam results ({', '.join(details)}) found. Sri Lankan IT companies typically don't require these.",
            "value": "Present (consider removing)", "score": 5, "evidence": evidence_lines,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0, "evidence": []}


def check_formatting_quality(pdf_path: str):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        issues: List[str] = []
        font_sizes: Set[float] = set()

        for page_num in range(len(doc)):
            for block in doc[page_num].get_text("dict").get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            try:
                                font_sizes.add(round(float(span.get("size", 0)), 1))
                            except Exception:
                                pass

        if len(font_sizes) < 2:
            issues.append("Use different font sizes for headings and body text")
        elif len(font_sizes) > 6:
            issues.append("Too many different font sizes - keep it consistent")

        blocks = doc[0].get_text("blocks")
        if blocks and min(b[0] for b in blocks) < 36:
            issues.append("Margins appear too small")

        score = max(0, min(10, 10 - len(issues) * 2))
        if issues:
            return {"status": "warning", "message": f"Formatting suggestions: {'; '.join(issues)}",
                    "value": f"{score}/10", "score": score}
        return {"status": "success", "message": "Formatting looks good.", "value": "Good ✓", "score": 10}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 5}
    finally:
        if doc:
            doc.close()


def validate_technical_keywords(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text = re.sub(r"\s+", " ", full_text).lower()

        found: List[str] = []
        for label in SRI_LANKAN_TECH_KEYWORDS:
            patterns = TECH_KEYWORD_VARIANTS.get(label, [])
            for pat in patterns:
                if re.search(pat, text, flags=re.IGNORECASE):
                    found.append(label)
                    break

        found = list(dict.fromkeys(found))
        kw_count = len(found)

        if kw_count >= 10:
            status, msg, score = "success", f"Excellent! Found {kw_count} relevant technical keywords.", 10
        elif kw_count >= 5:
            status, msg, score = "success", f"Good! Found {kw_count} technical keywords. Consider adding more.", 7
        elif kw_count >= 3:
            status, msg, score = "warning", f"Basic technical keywords found ({kw_count}). Add more tools/frameworks.", 5
        else:
            status, msg, score = "warning", f"Only {kw_count} technical keywords found. Add a clear skills section.", 3

        return {"status": status, "message": msg, "value": f"{kw_count} keywords",
                "found_keywords": found[:12], "score": score}
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
    bad = {"followers","following","repositories","repos","stars","starred",
           "packages","projects","settings"}
    return repo.lower() not in bad


def github_url_exists(url: str) -> Tuple[bool, str]:
    try:
        u = normalize_github_url(url)
        host = (urlparse(u).hostname or "").lower()
        if host not in {"github.com", "www.github.com"}:
            return False, "Blocked (non-GitHub host)"
        r = requests.get(u, timeout=HTTP_TIMEOUT, allow_redirects=True, headers=DEFAULT_UA)
        if r.status_code == 200:
            return True, "Working"
        if r.status_code == 404:
            return False, "Not found"
        return False, f"HTTP {r.status_code}"
    except Exception:
        return False, "Connection error"


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


def validate_github_links(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        links = [u for u in (bundle["links"] if bundle else []) if "github.com" in u.lower()]
        if not links:
            md    = get_markdown_cached(pdf_path, bundle) if bundle else pymupdf4llm.to_markdown(pdf_path)
            links = extract_github_links_from_text(md)

        links = list(dict.fromkeys(normalize_github_url(u) for u in links))
        if not links:
            return {"status": "warning", "message": "No GitHub links found.",
                    "repos": [], "value": "No links", "score": 3}

        repo_links    = [u for u in links if is_github_repo_link(u)]
        profile_links = [u for u in links if u not in repo_links]
        details: List[Dict] = []
        valid_profile = valid_repo = 0

        for u in profile_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "profile", "valid": ok, "message": msg})
            if ok: valid_profile += 1
            time.sleep(0.10)

        for u in repo_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "repo", "valid": ok, "message": msg})
            if ok: valid_repo += 1
            time.sleep(0.10)

        if valid_profile == 0 and valid_repo == 0:
            status, score = "warning", 3
        else:
            base  = 6 if valid_profile > 0 else 0
            boost = 4 if valid_repo >= 3 else 3 if valid_repo == 2 else 2 if valid_repo == 1 else 0
            score  = min(10, base + boost)
            status = "success" if score >= 7 else "warning"

        return {
            "status": status,
            "message": (f"GitHub found: {len(profile_links)} profile link(s), {len(repo_links)} repo link(s). "
                        f"Working: {valid_profile} profile, {valid_repo} repo."),
            "repos": details, "value": f"{valid_profile} profile, {valid_repo} repo working", "score": score,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "repos": [], "value": "Error", "score": 0}


def find_specialization(pdf_path: str, bundle: Dict[str, Any] = None):
    specializations = {
        "Software Technology":   "software technology",
        "Network Technology":    "network technology",
        "Multimedia Technology": "multimedia technology",
    }
    try:
        text = (bundle["text_lower"] if bundle else
                "\n".join([p.get_text().lower() for p in pymupdf.open(pdf_path)]))[:12000]
        for spec_name, keyword in specializations.items():
            if keyword in text:
                return {"status": "success", "message": f"Specialization: {spec_name}",
                        "value": spec_name, "score": 10}
        return {"status": "warning", "message": "Specialization area is not clearly mentioned.",
                "value": "Not specified", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_skills_separation(pdf_path: str):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        all_blocks: List = []
        for page in doc:
            all_blocks.extend(page.get_text("blocks"))
        all_blocks.sort(key=lambda b: (b[1], b[0]))

        sorted_lines = [
            line.strip()
            for block in all_blocks
            for line in block[4].splitlines()
        ]

        soft_skills: List[str]      = []
        technical_skills: List[str] = []
        current_section: Optional[str] = None

        soft_headers = ["soft skill", "soft skills", "interpersonal", "personal skills"]
        tech_headers = ["technical skill", "technical skills", "tech stack", "technologies", "programming"]
        stop_words   = [
            "project","experience","education","qualification","reference","contact",
            "certification","certificate","achievement","profile","summary","declaration",
            "language","interest","volunteer",
        ]

        for line in sorted_lines:
            ll = line.lower()
            if not line:
                continue
            if any(h in ll for h in soft_headers) and len(line) < 30:
                current_section = "soft"; continue
            if any(h in ll for h in tech_headers) and len(line) < 30:
                current_section = "technical"; continue
            if any(w in ll for w in stop_words) and len(line) < 30 and "skill" not in ll:
                current_section = None; continue

            if current_section == "soft" and len(line) > 2 and "skill" not in ll:
                soft_skills.append(line)
            elif current_section == "technical" and len(line) > 1 and "skill" not in ll:
                technical_skills.append(line)

        soft_skills      = list(dict.fromkeys(soft_skills))
        technical_skills = list(dict.fromkeys(technical_skills))

        if soft_skills and technical_skills:
            return {"status": "success", "message": "Both Technical and Soft Skills sections found.",
                    "value": "Both present ✓", "score": 10,
                    "soft_count": len(soft_skills), "tech_count": len(technical_skills)}
        if technical_skills:
            return {"status": "warning", "message": "Technical skills found, but Soft Skills section missing.",
                    "value": "Partial", "score": 6,
                    "tech_count": len(technical_skills), "soft_count": 0}
        if soft_skills:
            return {"status": "warning", "message": "Soft Skills found, but Technical Skills section missing.",
                    "value": "Partial", "score": 5,
                    "soft_count": len(soft_skills), "tech_count": 0}
        return {"status": "error", "message": "Could not identify clear skill sections.",
                "value": "Not found", "score": 2, "soft_count": 0, "tech_count": 0}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc:
            doc.close()


def check_contact_information(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text  = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text_lower = full_text.lower()

        has_phone    = bool(re.search(r"(\+94|0)?[\s-]?[0-9]{9,10}", full_text))
        has_email    = bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", full_text))
        has_linkedin = "linkedin" in text_lower
        has_location = any(w in text_lower for w in ["colombo", "sri lanka", "address", "location"])

        score = 0
        found: List[str] = []
        missing: List[str] = []

        for flag, pts, label in [
            (has_phone,    3, "Phone"),
            (has_email,    3, "Email"),
            (has_linkedin, 2, "LinkedIn"),
            (has_location, 2, "Location"),
        ]:
            if flag: score += pts; found.append(label)
            else:    missing.append(label)

        if score >= 8:
            return {"status": "success", "message": f"Complete contact info: {', '.join(found)}",
                    "value": "Complete ✓", "score": 10, "details": found}
        if score >= 6:
            return {"status": "success", "message": f"Good contact info. Found: {', '.join(found)}",
                    "value": "Good", "score": 8, "details": found}
        return {"status": "warning",
                "message": f"Missing: {', '.join(missing)}. Add complete contact information.",
                "value": "Incomplete", "score": max(4, score), "details": found}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_action_verbs(pdf_path: str, bundle: Dict[str, Any] = None):
    strong_verbs = [
        "developed","designed","implemented","created","built","led","managed",
        "optimized","improved","analyzed","achieved","delivered","collaborated",
        "architected","engineered","deployed","integrated","automated","streamlined",
    ]
    try:
        text_lower = (bundle["text_lower"] if bundle else
                      "\n".join([p.get_text().lower() for p in pymupdf.open(pdf_path)]))
        found_verbs = [v for v in strong_verbs if v in text_lower]
        n = len(found_verbs)

        if n >= 8:
            return {"status": "success", "message": f"Excellent! Found {n} strong action verbs.",
                    "value": f"{n} verbs ✓", "score": 10, "verbs": found_verbs[:10]}
        if n >= 5:
            return {"status": "success", "message": f"Good use of action verbs ({n} found).",
                    "value": f"{n} verbs", "score": 7, "verbs": found_verbs[:10]}
        if n >= 3:
            return {"status": "warning", "message": f"Only {n} action verbs found. Use more impact words.",
                    "value": f"{n} verbs", "score": 5, "verbs": found_verbs[:10]}
        return {"status": "warning",
                "message": "Very few action verbs. Use words like 'developed', 'implemented', 'designed'.",
                "value": "Weak", "score": 3, "verbs": found_verbs[:10]}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_quantifiable_achievements(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        patterns = [
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
            return {"status": "success", "message": f"Excellent! Found {n} quantifiable achievements.",
                    "value": f"{n} metrics ✓", "score": 10, "examples": hits[:5]}
        if n >= 3:
            return {"status": "success", "message": f"Good quantification with {n} metrics.",
                    "value": f"{n} metrics", "score": 7, "examples": hits[:5]}
        if n >= 1:
            return {"status": "warning", "message": f"Only {n} metrics found. Add more numbers to show impact.",
                    "value": f"{n} metrics", "score": 5, "examples": hits[:5]}
        return {"status": "warning",
                "message": "No quantifiable achievements. Add numbers (e.g., '40% improvement', '5 projects').",
                "value": "None found", "score": 2, "examples": []}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_professional_summary(pdf_path: str, bundle: Dict[str, Any] = None):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        first_page = doc[0]
        text = first_page.get_text().lower()
        keywords = ["summary","profile","objective","career objective",
                    "professional summary","about me","introduction"]
        has_summary = any(k in text for k in keywords)
        blocks = first_page.get_text("blocks")

        if blocks:
            top_section = first_page.rect.height * 0.3
            has_top = any(
                block[1] < top_section and any(k in (block[4] or "").lower() for k in keywords)
                for block in blocks
            )
            if has_top:
                return {"status": "success", "message": "Professional summary found at top of CV.",
                        "value": "Present ✓", "score": 10}
            if has_summary:
                return {"status": "success", "message": "Summary section found.",
                        "value": "Present", "score": 8}
            return {"status": "info", "message": "No professional summary. Consider adding a brief career objective.",
                    "value": "Not found", "score": 6}
        return {"status": "warning", "message": "Could not detect professional summary.",
                "value": "Not detected", "score": 5}
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
        adjusted["photo"] = {**adjusted["photo"],
                             "message": "Blind mode: photo ignored for fairness.",
                             "score": 10, "status": "info"}
    if "ol_al" in adjusted:
        adjusted["ol_al"] = {**adjusted["ol_al"],
                             "message": "Blind mode: school results treated as low importance.",
                             "score": max(adjusted["ol_al"].get("score", 0), 7)}
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
def validate_with_llm(pdf_path: str, validation_results: Dict[str, Any],
                       bundle: Dict[str, Any], temperature: float = 0.1):
    try:
        markdown_text = get_markdown_cached(pdf_path, bundle)
        groq_api_key  = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            return {"status": "error", "message": "API key missing.",
                    "results": [], "passed": 0, "total": 9, "score": 0}

        query = f"""
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
        client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )

        raw     = (response.choices[0].message.content or "").strip()
        data    = safe_json_loads(raw)
        answers = data.get("answers", [])
        total   = 9
        passed  = sum(1 for a in answers[:total] if a.get("yes") is True)
        score   = round((passed / total) * 10, 1)

        return {"status": "success", "message": "AI done",
                "results": answers[:total], "passed": passed, "total": total, "score": score}
    except Exception as e:
        return {"status": "error", "message": f"AI Analysis error: {str(e)}",
                "results": [], "passed": 0, "total": 9, "score": 0}


# ──────────────────────────────────────────────────────────────
# Overall score
# ──────────────────────────────────────────────────────────────
def calculate_overall_score(results: Dict[str, Any], llm_std: float = 0.0):
    weights = {
        "page_count": 0.05, "gpa": 0.05, "professional_email": 0.08,
        "photo": 0.04, "ol_al": 0.05, "formatting": 0.06,
        "specialization": 0.04, "github": 0.08, "skills": 0.08,
        "keywords": 0.10, "contact": 0.06, "action_verbs": 0.07,
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
        grade, status = "Excellent", "success"
    elif final >= 7.0:
        grade, status = "Good", "success"
    elif final >= 5.5:
        grade, status = "Fair", "warning"
    else:
        grade, status = "Needs Improvement", "warning"

    return {
        "status": status, "score": round(final, 1), "grade": grade,
        "message": f"Overall CV Score: {round(final, 1)}/10 - {grade}",
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
    impact_10  = _weighted_avg_10([(_get_score10(results,"achievements"), 0.45),
                                   (_get_score10(results,"action_verbs"),  0.35),
                                   (_get_score10(results,"summary"),       0.20)])
    brevity_10 = _weighted_avg_10([(_get_score10(results,"page_count"), 0.75),
                                   (_get_score10(results,"ol_al"),      0.25)])
    style_10   = _weighted_avg_10([(_get_score10(results,"formatting"),         0.70),
                                   (_get_score10(results,"professional_email"), 0.20),
                                   (_get_score10(results,"photo"),              0.10)])
    skills_10  = _weighted_avg_10([(_get_score10(results,"skills"),   0.40),
                                   (_get_score10(results,"keywords"),  0.40),
                                   (_get_score10(results,"github"),    0.20)])

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
def get_time_greeting(now: datetime = None) -> str:
    h = (now or datetime.now()).hour
    if   5  <= h < 12: return "Good morning."
    elif 12 <= h < 17: return "Good afternoon."
    elif 17 <= h < 22: return "Good evening."
    else:              return "Good night."


# ──────────────────────────────────────────────────────────────
# CACHE helpers (atomic write)
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
def _get_text_from_selectors(soup: BeautifulSoup, selectors: list) -> str:
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
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()
    best = ""
    for block in soup.find_all(["div", "section", "article"]):
        txt = block.get_text(separator=" ", strip=True)
        if len(txt) > len(best):
            best = txt
    return best[:2000] if best else ""


def scrape_job_page(url: str, session_obj: requests.Session) -> Tuple[str, str]:
    if not url or not url.startswith("http"):
        return "", ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session_obj.get(url, headers=SCRAPER_HEADERS,
                                   timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code == 404:
                return "", ""
            if resp.status_code != 200:
                time.sleep(REQUEST_DELAY[1])
                continue

            soup        = BeautifulSoup(resp.text, "html.parser")
            description = _get_text_from_selectors(soup, DESCRIPTION_SELECTORS)
            raw_skills  = _get_text_from_selectors(soup, SKILLS_SELECTORS) or _get_skill_tags(soup)
            if not description:
                description = _fallback_body_text(soup)
            return description.strip(), raw_skills.strip()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(REQUEST_DELAY[1] * attempt)

    return "", ""


def enrich_csv_with_descriptions(
    csv_path: str = CSV_PATH,
    output_path: Optional[str] = None,
    cache_path: str = CACHE_CSV,
    force_rescrape: bool = False,
) -> pd.DataFrame:
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = base + "_enriched" + ext

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "url" not in df.columns:
        logger.error("No 'url' column — cannot scrape.")
        return df

    for col in ("description", "raw_skills"):
        if col not in df.columns:
            df[col] = ""

    cache = load_cache(cache_path)

    def _needs_scrape(row: pd.Series) -> bool:
        if force_rescrape:
            return True
        desc = str(row.get("description", "")).strip()
        return not desc or bool(_GENERIC_DESC_RE.search(desc)) or len(desc) < 40

    to_scrape = df[df.apply(_needs_scrape, axis=1)]
    if to_scrape.empty:
        _safe_to_csv(df, output_path)
        return df

    session = requests.Session()
    scraped = cache_hits = 0

    for idx, row in to_scrape.iterrows():
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        if not force_rescrape and url in cache:
            df.at[idx, "description"], df.at[idx, "raw_skills"] = cache[url]
            cache_hits += 1
            continue

        desc, skills = scrape_job_page(url, session)
        if desc:   df.at[idx, "description"] = desc
        if skills: df.at[idx, "raw_skills"]  = skills
        cache[url] = (df.at[idx, "description"], df.at[idx, "raw_skills"])
        scraped += 1

        if scraped % 10 == 0:
            save_cache(cache, cache_path)
        time.sleep(random.uniform(*REQUEST_DELAY))

    session.close()
    save_cache(cache, cache_path)
    _safe_to_csv(df, output_path)
    logger.info(f"Enrichment done — scraped: {scraped}, cache hits: {cache_hits}, saved: {output_path}")
    return df


# ──────────────────────────────────────────────────────────────
# Build combined fields for TF-IDF
# ──────────────────────────────────────────────────────────────
def build_combined_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "title": "Not specified", "company": "Company Name Withheld",
        "location": "Sri Lanka", "description": "", "raw_skills": "",
        "closing_date": "", "url": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    def clean_desc(d: str) -> str:
        d = d.strip()
        return "" if (not d or (_GENERIC_DESC_RE.search(d) and len(d) < 120)) else d

    df["desc_clean"] = df["description"].apply(clean_desc)

    combined = df["title"] + " " + df["desc_clean"] + " " + df["raw_skills"]
    df["job_skill_list"] = combined.apply(extract_skills_from_text)
    df["inferred_skills"] = df["title"].apply(infer_skills_from_title)

    # Fill sparse skill lists with inferred skills, then clean
    df["job_skill_list"] = df.apply(
        lambda r: list(dict.fromkeys((r["job_skill_list"] or []) + (r["inferred_skills"] or [])))
        if len(r["job_skill_list"] or []) < 3 else (r["job_skill_list"] or []),
        axis=1,
    )
    df["job_skill_list"] = df["job_skill_list"].apply(clean_skill_list)
    df["job_skill_text"] = df["job_skill_list"].apply(lambda xs: " ".join(xs or []))

    df["job_text"] = (
        df["title"] + " " + df["company"] + " " +
        df["desc_clean"] + " " + df["raw_skills"] + " " + df["job_skill_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# ──────────────────────────────────────────────────────────────
# Load job data
# ──────────────────────────────────────────────────────────────
def load_job_data(csv_path: str = CSV_PATH, auto_enrich: bool = True) -> Optional[pd.DataFrame]:
    try:
        if auto_enrich:
            base, ext = os.path.splitext(csv_path)
            enriched_path = base + "_enriched" + ext
            df = (pd.read_csv(enriched_path, dtype=str).fillna("")
                  if os.path.exists(enriched_path)
                  else enrich_csv_with_descriptions(csv_path=csv_path))
        else:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        return build_combined_fields(df)
    except Exception as e:
        logger.error(f"Error loading job data: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Date filter
# ──────────────────────────────────────────────────────────────
def _parse_topjobs_date_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce", format="%a %b %d %Y")
    if dt.notna().any():
        return dt
    return pd.to_datetime(s, errors="coerce")


def filter_active_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "closing_date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["closing_dt"] = _parse_topjobs_date_series(out["closing_date"].astype(str))
    today = pd.Timestamp.today().normalize()
    return out[out["closing_dt"].notna() & (out["closing_dt"] >= today)]


# ──────────────────────────────────────────────────────────────
# User skills from CV
# ──────────────────────────────────────────────────────────────
def build_user_skills_from_cv(results: Dict[str, Any], cv_text: str) -> List[str]:
    skills: List[str] = []

    # From keyword validator
    kw = (results or {}).get("keywords", {}).get("found_keywords", [])
    if isinstance(kw, list):
        skills.extend(normalize_token(x) for x in kw)

    # From raw CV text
    skills.extend(extract_skills_from_text(cv_text or ""))

    # From GitHub evidence text
    try:
        gh_msg = (results or {}).get("github", {}).get("message", "") or ""
        skills.extend(extract_skills_from_text(gh_msg))
    except Exception:
        pass

    return clean_skill_list(skills)  # dedupe + drop non-tech words


# ──────────────────────────────────────────────────────────────
# MAIN RECOMMENDER
# ──────────────────────────────────────────────────────────────
_LEVEL_HINT = {
    "intern_junior": "intern junior trainee entry level graduate",
    "mid":           "mid level experienced engineer developer",
    "senior":        "senior lead architect principal manager",
}


def get_job_recommendations(
    results: Dict[str, Any],
    cv_text: str,
    location_filter: str = "All",
    top_n: int = 5,
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

        # location filter
        if location_filter and location_filter != "All":
            df = df[df["location"].str.contains(location_filter, case=False, na=False)]
            if df.empty:
                return []

        # active jobs only
        df = filter_active_jobs(df)
        if df.empty:
            return []

        # seniority
        cv_level = estimate_cv_level(cv_text)
        df = apply_seniority_handling(df, cv_level=cv_level, mode=seniority_mode)
        if df.empty:
            return []

        # build TF-IDF query
        level_hint = _LEVEL_HINT.get(cv_level, "junior entry level")
        query_text = (level_hint + " " + " ".join(user_skills)).strip()

        job_texts = df["job_text"].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english",
                              sublinear_tf=True)  # sublinear_tf helps sparse skill matches
        mat = vec.fit_transform(job_texts + [query_text])
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()

        df = df.reset_index(drop=True)
        df["final_score"] = sims

        # ── Skill-overlap boost ──────────────────────────────────
        # Add a small bonus proportional to how many user skills appear
        # in the job's skill list — this improves relevance when TF-IDF
        # scores are otherwise close.
        cv_set: Set[str] = set(user_skills)

        def _overlap_ratio(job_skills: Any) -> float:
            js = set(clean_skill_list(job_skills or []))
            if not js:
                return 0.0
            return len(js & cv_set) / max(len(js), len(cv_set))

        df["overlap"] = df["job_skill_list"].apply(_overlap_ratio)
        # Weighted combination: 80% TF-IDF + 20% skill overlap
        df["final_score"] = df["final_score"] * 0.80 + df["overlap"] * 0.20
        # ─────────────────────────────────────────────────────────

        df = (df.sort_values("final_score", ascending=False)
               .drop_duplicates(subset=["title", "company"], keep="first")
               .head(int(top_n)))

        jobs: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            job_skills = clean_skill_list(row.get("job_skill_list") or [])
            job_set    = set(job_skills)
            matched    = sorted(job_set & cv_set)
            missing    = sorted(job_set - cv_set)

            raw_desc  = str(row.get("description", ""))
            desc_short = (raw_desc[:220] + "...") if len(raw_desc) > 220 else raw_desc

            match_pct = round(min(100.0, max(0.0, float(row["final_score"]) * 100.0)), 1)

            jobs.append({
                "title":            row.get("title", ""),
                "company":          row.get("company", ""),
                "location":         row.get("location", ""),
                "description":      desc_short or "Please refer the vacancy.",
                "url":              row.get("url", ""),
                "closing_date":     row.get("closing_date", ""),
                "match_percentage": match_pct,
                "match_level":      ("Excellent" if match_pct >= 70
                                     else "Good" if match_pct >= 50
                                     else "Potential"),
                "cv_level_used":    cv_level,
                "matched_skills":   matched[:8],
                "missing_skills":   missing[:8],
            })

        results["matching_skills_used"] = user_skills
        return jobs

    except Exception as e:
        logger.exception(f"Error in get_job_recommendations: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────
@app.route("/stats")
def stats():
    stats_data = fetch_stats()
    return render_template("stats.html", stats=stats_data)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cleanup_old_previews()

        if "cv_file" not in request.files:
            return "Please select a CV file."
        file = request.files["cv_file"]
        if file.filename == "":
            return "Please select a CV file."
        if not is_probably_pdf(file):
            return "Please upload a valid PDF file only."

        blind_mode = request.form.get("blind_mode", "off") == "on"
        filename   = secure_filename(file.filename)
        temp_name  = f"{int(time.time())}_{filename}"
        filepath   = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
        file.save(filepath)

        results         = None
        recommendations = []
        greeting        = get_time_greeting()
        sub_greeting    = "Welcome to your resume review."
        preview_info    = {"preview_id": "", "preview_filename": ""}

        try:
            preview_info = save_preview_file(filepath, filename)
            file_hash    = sha256_file(filepath)
            bundle       = extract_pdf_bundle(filepath)

            results = {
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
                location_filter=request.form.get("location_filter", "All"),
                top_n=1000,
                seniority_mode="filter",
            )

            session["job_recommendations"] = recommendations
            session["last_filename"]       = filename

            try:
                save_resume_run(
                    file_hash=file_hash, filename=filename,
                    overall_score=results["overall"]["score"],
                    grade=results["overall"]["grade"],
                    llm_score=results["llm"].get("score", 0),
                    llm_std=0.0,
                )
            except Exception:
                pass

        finally:
            gc.collect()
            time.sleep(0.05)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

        return render_template(
            "index.html",
            results=results,
            filename=filename,
            greeting=greeting,
            sub_greeting=sub_greeting,
            job_recommendations=recommendations,
        )

    return render_template("index.html", results=None)


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

    blind_mode = request.form.get("blind_mode", "off") == "on"
    filename   = secure_filename(file.filename)
    temp_name  = f"{int(time.time())}_{filename}"
    filepath   = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
    file.save(filepath)

    try:
        preview_info = save_preview_file(filepath, filename)
        bundle       = extract_pdf_bundle(filepath)

        results = {
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
        results["job_recommendations"] = get_job_recommendations(
            results=results,
            cv_text=bundle.get("text", ""),
            location_filter=request.form.get("location_filter", "All"),
            top_n=int(request.form.get("top_n", 10000)),
            seniority_mode=request.form.get("seniority_mode", "filter"),
        )
        return jsonify(results)
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@app.route("/job-recommendations", endpoint="job_recommendations")
def job_recommendations_page():
    jobs     = session.get("job_recommendations", [])
    filename = session.get("last_filename", "")
    return render_template("job_recommendation.html",
                           results={"job_recommendations": jobs}, filename=filename)


@app.route("/preview/<preview_id>/<filename>")
def preview_file(preview_id: str, filename: str):
    preview_id = re.sub(r"[^a-f0-9]", "", preview_id.lower())
    filename   = secure_filename(filename)
    user_dir   = os.path.join(PREVIEW_FOLDER, preview_id)
    if not preview_id or not filename or not os.path.exists(os.path.join(user_dir, filename)):
        abort(404)
    return send_from_directory(user_dir, filename, mimetype="application/pdf")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Validator App / TopJobs enricher")
    parser.add_argument("--enrich", action="store_true", help="Run CSV enrichment then exit")
    parser.add_argument("--csv",    default=CSV_PATH,    help="Input CSV path")
    parser.add_argument("--cache",  default=CACHE_CSV,   help="Cache CSV path")
    parser.add_argument("--force",  action="store_true", help="Force re-scrape all URLs")
    args = parser.parse_args()

    if args.enrich:
        enrich_csv_with_descriptions(
            csv_path=args.csv, cache_path=args.cache, force_rescrape=args.force,
        )
        print("Enrichment complete.")
    else:
        print("\n" + "=" * 55)
        print("  Enhanced CV Validator for Sri Lankan IT Market")
        print("  http://127.0.0.1:5000")
        print("=" * 55 + "\n")
        app.run(debug=True, host="0.0.0.0", port=5000)