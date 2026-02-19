from flask import Flask, request, render_template, jsonify, abort, send_from_directory, url_for
import os
import re
import time
import gc
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import json
import uuid
import shutil

import requests
import pymupdf  # PyMuPDF
import pymupdf4llm
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
from openai import OpenAI

# For job recommendations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config
# -----------------------------
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

BASE_DIR = os.path.dirname(__file__)
PREVIEW_FOLDER = os.path.join(BASE_DIR, "previews")
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

HTTP_TIMEOUT = 10
DEFAULT_UA = {"User-Agent": "Mozilla/5.0"}

# Preview retention (seconds). Old previews will be deleted.
PREVIEW_TTL_SECONDS = 30 * 60  # 30 minutes

# -----------------------------
# SQLite
# -----------------------------
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
    file_hash: str, filename: str, overall_score: float, grade: str, llm_score: float, llm_std: float
):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resume_runs (file_hash, filename, overall_score, grade, llm_score, llm_std, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_hash, filename, overall_score, grade, llm_score, llm_std, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


init_db()

# -----------------------------
# Tech keyword variants (improved matching)
# -----------------------------
TECH_KEYWORD_VARIANTS = {
    "Java": [r"\bjava\b(?!\s*script)"],
    "Python": [r"\bpython\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "C#": [r"\bc\s*#\b", r"\bcsharp\b", r"\bc\s*sharp\b"],
    "C++": [r"\bc\+\+\b"],
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgres(?:ql)?\b"],
    "MongoDB": [r"\bmongo(?:db)?\b"],
    "Oracle": [r"\boracle\b"],
    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "GitLab": [r"\bgitlab\b"],
    "HTML5": [r"\bhtml5\b", r"\bhtml\b"],
    "CSS3": [r"\bcss3\b", r"\bcss\b"],
    "React": [r"\breact(?:\.js|js)?\b"],
    "Angular": [r"\bangular\b", r"\bangularjs\b"],
    "Vue.js": [r"\bvue(?:\.js|js)?\b"],
    "Node.js": [r"\bnode(?:\.js|js)?\b"],
    "Express": [r"\bexpress(?:\.js|js)?\b"],
    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "Spring Boot": [r"\bspring\s*boot\b", r"\bspringboot\b"],
    "Android": [r"\bandroid\b"],
    "iOS": [r"\bios\b"],
    "Flutter": [r"\bflutter\b"],
    "React Native": [r"\breact\s*native\b"],
    "AWS": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "Azure": [r"\bazure\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "REST API": [r"\brest(?:ful)?\s*api(?:s)?\b", r"\brest\b"],
    "GraphQL": [r"\bgraphql\b"],
    "Microservices": [r"\bmicroservices?\b"],
    "Agile": [r"\bagile\b"],
    "Scrum": [r"\bscrum\b"],
    "JIRA": [r"\bjira\b"],
}

# For UI display ordering
SRI_LANKAN_TECH_KEYWORDS = list(TECH_KEYWORD_VARIANTS.keys())

# Soft skills that employers look for
VALUED_SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "adaptability",
    "collaboration", "creativity", "attention to detail",
    "work ethic", "interpersonal"
]

COMMON_MISTAKES = {
    "ol_al_present": "School exam results (O/L, A/L) are typically not needed for IT jobs",
    "unprofessional_email": "Email address should be professional (firstname.lastname@domain.com)",
    "no_photo": "Professional photo is recommended for Sri Lankan CVs",
    "poor_formatting": "CV formatting needs improvement for better readability",
    "missing_tech_skills": "Technical skills section is not clearly visible",
    "weak_projects": "Projects should clearly mention technologies used",
    "no_github": "GitHub profile/projects links are missing",
    "wrong_length": "CV should be 1-2 pages for internship/junior positions"
}

# -----------------------------
# Safety helpers
# -----------------------------
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


# -----------------------------
# Preview helpers (safer than base64)
# -----------------------------
def cleanup_old_previews(ttl_seconds: int = PREVIEW_TTL_SECONDS):
    """Delete preview folders older than TTL."""
    now = time.time()
    try:
        for name in os.listdir(PREVIEW_FOLDER):
            d = os.path.join(PREVIEW_FOLDER, name)
            if not os.path.isdir(d):
                continue
            try:
                mtime = os.path.getmtime(d)
                if now - mtime > ttl_seconds:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def save_preview_file(src_pdf_path: str, original_filename: str) -> Dict[str, str]:
    """Copy uploaded PDF into previews/<preview_id>/<filename> and return ids."""
    preview_id = uuid.uuid4().hex
    safe_name = secure_filename(original_filename) or "resume.pdf"

    dest_dir = os.path.join(PREVIEW_FOLDER, preview_id)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, safe_name)
    shutil.copyfile(src_pdf_path, dest_path)

    return {"preview_id": preview_id, "preview_filename": safe_name}


# -----------------------------
# PDF extraction bundle
# -----------------------------
def extract_pdf_bundle(pdf_path: str) -> Dict[str, Any]:
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = []
        has_image = False
        links = []

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
            "text": text,
            "text_lower": text.lower(),
            "has_image": has_image,
            "links": list(dict.fromkeys(links)),
            "page_count": len(doc),
            "markdown": None,
        }
    finally:
        if doc:
            doc.close()


def get_markdown_cached(pdf_path: str, bundle: Dict[str, Any]) -> str:
    if bundle.get("markdown") is None:
        bundle["markdown"] = pymupdf4llm.to_markdown(pdf_path)
    return bundle["markdown"]


# -----------------------------
# Evidence helper
# -----------------------------
def find_evidence_snippets(text: str, patterns: List[str], max_hits: int = 3, window: int = 80) -> List[str]:
    if not text:
        return []
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            snippet = text[start:end].replace("\n", " ").strip()
            out.append(snippet)
            if len(out) >= max_hits:
                return out
    return out


# -----------------------------
# ============================================================
# JOB RECOMMENDATION 
# ============================================================

from typing import List, Dict, Any, Set
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Helpers: Seniority
# -----------------------------
SENIOR_TITLE_RE = re.compile(
    r"\b(senior|lead|manager|head|architect|principal|tech lead|team lead|director)\b",
    re.IGNORECASE
)

JUNIOR_HINTS_RE = re.compile(
    r"\b(intern|internship|trainee|undergraduate|student|fresh(er)?|entry[-\s]*level|junior)\b",
    re.IGNORECASE
)

YEARS_RE = re.compile(r"(\d+)\+?\s*years", re.IGNORECASE)

def estimate_cv_level(cv_text: str) -> str:
    t = (cv_text or "").strip()
    if not t:
        return "junior"
    if JUNIOR_HINTS_RE.search(t):
        return "junior"
    years = [int(x) for x in YEARS_RE.findall(t)]
    y = max(years) if years else 0
    if y >= 5:
        return "senior"
    if y >= 2:
        return "mid"
    return "junior"

def apply_seniority_handling(df: pd.DataFrame, cv_level: str, mode: str = "filter") -> pd.DataFrame:
    df = df.copy()
    is_senior_job = df["title"].fillna("").astype(str).str.contains(SENIOR_TITLE_RE)
    if mode == "filter":
        if cv_level in ("junior", "mid"):
            df = df[~is_senior_job]
        return df

    # penalty mode
    if "final_score" in df.columns:
        if cv_level == "junior":
            df.loc[is_senior_job, "final_score"] *= 0.55
        elif cv_level == "mid":
            df.loc[is_senior_job, "final_score"] *= 0.85
    return df

# -----------------------------
# Date parsing + active jobs
# -----------------------------
def _parse_topjobs_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="%a %b %d %Y")

def filter_active_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "closing_date" not in df.columns:
        return df.copy()

    today = pd.Timestamp.today().normalize()
    df = df.copy()
    df["closing_dt"] = _parse_topjobs_date_series(df["closing_date"].astype(str))
    df = df[df["closing_dt"].notna() & (df["closing_dt"] >= today)]
    return df

# -----------------------------
# Normalization (skills)
# -----------------------------
_SKILL_SYNONYMS = {
    "js": "javascript",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "expressjs": "express",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "rest": "rest api",
    "api": "rest api",
    "k8s": "kubernetes",
}

def normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return _SKILL_SYNONYMS.get(t, t)

def normalize_keywords(items: List[str]) -> List[str]:
    out: List[str] = []
    for k in items or []:
        nk = normalize_token(k)
        if nk:
            out.append(nk)
    return list(dict.fromkeys(out))

# -----------------------------
# Extract skills from text using TECH_KEYWORD_VARIANTS
# (depends on your existing TECH_KEYWORD_VARIANTS dict)
# -----------------------------
def extract_job_skills(text: str) -> List[str]:
    if not text:
        return []
    t = " " + re.sub(r"\s+", " ", text).lower() + " "
    found: List[str] = []
    for label, patterns in TECH_KEYWORD_VARIANTS.items():
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                found.append(normalize_token(label))
                break
    return list(dict.fromkeys(found))

# -----------------------------
# Combine job fields
# -----------------------------
def build_combined_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df.get("title", "").fillna("Not specified").astype(str)
    df["company"] = df.get("company", "").fillna("Company Name Withheld").astype(str)
    df["location"] = df.get("location", "").fillna("Sri Lanka").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)

    generic_desc = re.compile(r"please refer (the )?(vacancy|advert|advertisement)", re.IGNORECASE)

    def clean_desc(d: str) -> str:
        d = (d or "").strip()
        if not d:
            return ""
        if generic_desc.search(d) and len(d) < 80:
            return ""
        return d

    df["desc_clean"] = df["description"].apply(clean_desc)
    df["title_text"] = (df["title"] + " " + df["company"]).str.replace(r"\s+", " ", regex=True).str.strip()
    df["desc_text"] = df["desc_clean"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["job_skill_list"] = (df["title"] + " " + df["desc_clean"]).apply(extract_job_skills)
    df["job_skill_text"] = df["job_skill_list"].apply(lambda xs: " ".join(xs))
    return df

def load_job_data(csv_path: str = "topjobs_it_jobs.csv") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        for col in ["title", "company", "location", "description", "closing_date", "url"]:
            if col not in df.columns:
                df[col] = ""
        return build_combined_fields(df)
    except Exception as e:
        print(f"Error loading job data: {e}")
        return None

# -----------------------------
# Scoring helpers
# -----------------------------
def _cosine_sim_for_fields(job_field_texts: List[str], query_text: str) -> List[float]:
    all_text = job_field_texts + [query_text]
    vec = TfidfVectorizer(max_features=1400, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(all_text)
    user_vec = mat[-1]
    job_vecs = mat[:-1]
    return cosine_similarity(user_vec, job_vecs).flatten().tolist()

def _skill_overlap_bonus(job_skills: List[str], cv_skills: List[str]) -> float:
    js = set(job_skills or [])
    cs = set(cv_skills or [])
    if not js or not cs:
        return 0.0
    overlap = len(js.intersection(cs))
    denom = max(1, len(js))
    ratio = overlap / denom
    return min(0.20, ratio * 0.20)

def _urgency_bonus(closing_dt, max_days: int = 14) -> float:
    try:
        if pd.isna(closing_dt):
            return 0.0
        today = pd.Timestamp.today().normalize()
        days_left = int((closing_dt.normalize() - today).days)
        if days_left < 0 or days_left > max_days:
            return 0.0
        return (1.0 - (days_left / max_days)) * 0.05
    except Exception:
        return 0.0

# -----------------------------
# Build user skills (NOVEL, but safe)
# -----------------------------
def build_user_skills_from_cv(results: Dict[str, Any], cv_text: str) -> List[str]:
    """
    Uses ONLY technical-ish sources:
      - keyword validator found_keywords
      - skills section tech_samples
      - extra scan for tech keywords in full CV text
    Does NOT change skill validation rules; this is only for job matching.
    """
    kw = results.get("keywords", {}).get("found_keywords", []) or []
    tech_samples = (results.get("skills", {}) or {}).get("tech_samples", []) or []
    extra = extract_job_skills(cv_text or "")

    merged = normalize_keywords(kw + tech_samples + extra)
    return merged

# -----------------------------
# MAIN: Job recommendations
# -----------------------------
def get_job_recommendations(
    results: Dict[str, Any],
    cv_text: str,
    location_filter: str = "All",
    top_n: int = 5,
    seniority_mode: str = "filter",  # "filter" or "penalty"
) -> List[Dict[str, Any]]:

    try:
        df = load_job_data()
        if df is None or df.empty:
            return []

        # Build better user skills list (keywords + skills section + extra scan)
        user_skills = build_user_skills_from_cv(results, cv_text=cv_text)
        if not user_skills:
            return []

        # Location filter
        if location_filter and location_filter != "All":
            df = df[df["location"].astype(str).str.contains(location_filter, case=False, na=False)]
        if df.empty:
            return []

        # Active jobs
        df = filter_active_jobs(df)
        if df.empty:
            return []

        # Seniority level
        cv_level = estimate_cv_level(cv_text)
        query_text = f"level {cv_level} skills " + " ".join(user_skills)

        # Similarities
        df = df.copy()
        df["sim_title"] = _cosine_sim_for_fields(df["title_text"].astype(str).tolist(), query_text)
        df["sim_desc"]  = _cosine_sim_for_fields(df["desc_text"].astype(str).tolist(), query_text)
        df["sim_skill"] = _cosine_sim_for_fields(df["job_skill_text"].astype(str).tolist(), " ".join(user_skills))

        # Weighted base score
        df["base_score"] = (0.45 * df["sim_title"]) + (0.25 * df["sim_desc"]) + (0.30 * df["sim_skill"])

        # Bonuses
        df["overlap_bonus"] = df["job_skill_list"].apply(lambda js: _skill_overlap_bonus(js, user_skills))
        df["urgency_bonus"] = df["closing_dt"].apply(_urgency_bonus)
        df["final_score"] = df["base_score"] + df["overlap_bonus"] + df["urgency_bonus"]

        # Seniority handling
        df = apply_seniority_handling(df, cv_level=cv_level, mode=seniority_mode)
        if df.empty:
            return []

        # Sort + dedupe
        df = df.sort_values("final_score", ascending=False)
        df = df.drop_duplicates(subset=["title", "company"], keep="first")
        recommendations = df.head(top_n)

        # Output (matched + missing)
        cv_set: Set[str] = set(user_skills)
        jobs: List[Dict[str, Any]] = []

        for _, row in recommendations.iterrows():
            job_skills = row.get("job_skill_list", []) or []
            job_set = set(job_skills)

            matched = sorted(job_set.intersection(cv_set))
            missing = sorted(job_set - cv_set)

            desc = row.get("description", "")
            desc_short = (desc[:220] + "...") if isinstance(desc, str) and len(desc) > 220 else desc

            match_percentage = float(row["final_score"]) * 100.0

            jobs.append({
                "title": row.get("title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "description": desc_short or "Please refer the vacancy",
                "url": row.get("url", ""),
                "closing_date": row.get("closing_date", ""),
                "match_percentage": round(match_percentage, 1),
                "match_level": "Excellent" if match_percentage >= 70 else "Good" if match_percentage >= 50 else "Potential",
                "cv_level_used": cv_level,
                "matched_skills": matched[:8],
                "missing_skills": missing[:8],
            })

        # OPTIONAL: store for UI ("Skills Used for Job Matching")
        results["matching_skills_used"] = user_skills

        return jobs

    except Exception as e:
        print(f"Error in job recommendations: {e}")
        return []

# -----------------------------
# CV Validation Functions
# -----------------------------
def check_cv_page_count(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        page_count = bundle["page_count"] if bundle else len(pymupdf.open(pdf_path))
        if page_count == 1:
            return {"status": "success", "message": "Perfect! Single page CV.", "value": "1 page ✓", "score": 10}
        elif page_count == 2:
            return {"status": "success", "message": "Good! Two pages is acceptable.", "value": "2 pages ✓", "score": 8}
        else:
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
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text_lower = full_text.lower()

        gpa_patterns = [
            r"(?:current\s*)?gpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"cgpa\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"grade\s*point\s*(?:average)?\s*[:\-]\s*(\d(?:\.\d{1,2})?)",
            r"gpa\s*(?:is)?\s*(\d(?:\.\d{1,2})?)",
        ]

        found_value = None
        evidence_line = None
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

        for line in lines:
            for pat in gpa_patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    found_value = m.group(1)
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
                "message": "GPA keyword found, but GPA value is not clearly written. Use format like 'Current GPA: 3.71'.",
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


def check_professional_email(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])

        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = list(dict.fromkeys(re.findall(email_pattern, full_text)))

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

        # --- Soft guidance (NOT a strict judgement) ---
        local = emails[0].split("@")[0].lower()

        slang_words = {"cool","hot","sexy","boss","king","queen","swag","ninja","devil","angel","xoxo"}
        risk_points = 0
        reasons = []

        if any(w in local for w in slang_words):
            risk_points += 2
            reasons.append("contains informal word")

        if re.search(r"\d{4,}", local):
            risk_points += 1
            reasons.append("has long number sequence")

        if len(re.findall(r"[._-]", local)) > 3:
            risk_points += 1
            reasons.append("uses many symbols")

        if risk_points == 0:
            risk = "Low"
            score = 10
            msg = "Email found. Looks fine for professional use."
        elif risk_points <= 2:
            risk = "Medium"
            score = 8
            msg = "Email found. Consider a simpler name-based email for best impression."
        else:
            risk = "High"
            score = 6
            msg = "Email found, but it may look informal. A name-based email is recommended."

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


def check_photo_presence(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        has_image = bundle["has_image"] if bundle else False
        if bundle is None:
            doc = pymupdf.open(pdf_path)
            try:
                for i in range(len(doc)):
                    if doc[i].get_images(full=True):
                        has_image = True
                        break
            finally:
                doc.close()

        if has_image:
            return {
                "status": "success",
                "message": "Photo found ",
                "value": "Present ✓",
                "score": 10,
            }
        else:
            return {
                "status": "info",
                "message": "No photo detected. Consider adding a professional photo.",
                "value": "Not found",
                "score": 7,
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_ol_al_presence(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = (bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])) or ""
        text_lower = (bundle["text_lower"] if bundle else full_text.lower()) or full_text.lower()

        ol_patterns = ["o/l", "o.l", "ordinary level", "g.c.e o/l", "gce o/l", "ordinary level examination"]
        al_patterns = ["a/l", "a.l", "advanced level", "g.c.e a/l", "gce a/l", "advanced level examination"]

        has_ol = any(p in text_lower for p in ol_patterns)
        has_al = any(p in text_lower for p in al_patterns)

        if not (has_ol or has_al):
            return {
                "status": "success",
                "message": "No school exam results found. Good - focus on degree and technical skills.",
                "value": "Not present ✓",
                "score": 10,
                "evidence": [],
            }

        patterns = [
            r"\b(o\/l|o\.l|g\.c\.e\s*o\/l|gce\s*o\/l|ordinary level)\b",
            r"\b(a\/l|a\.l|g\.c\.e\s*a\/l|gce\s*a\/l|advanced level)\b",
        ]

        evidence_lines = []
        for line in full_text.splitlines():
            line_clean = " ".join(line.split())
            if len(line_clean) < 6:
                continue
            if any(re.search(p, line_clean, flags=re.IGNORECASE) for p in patterns):
                evidence_lines.append(line_clean)
            if len(evidence_lines) >= 3:
                break

        if not evidence_lines:
            evidence_lines = find_evidence_snippets(full_text, patterns, max_hits=2, window=50)

        details = []
        if has_ol:
            details.append("O/L")
        if has_al:
            details.append("A/L")

        return {
            "status": "warning",
            "message": f"School exam results ({', '.join(details)}) found. Sri Lankan IT companies typically don't require these for graduate positions.",
            "value": "Present (consider removing)",
            "score": 5,
            "evidence": evidence_lines,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0, "evidence": []}


def check_formatting_quality(pdf_path: str):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        issues = []
        font_sizes = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
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

        page = doc[0]
        blocks = page.get_text("blocks")
        if blocks:
            min_x = min(b[0] for b in blocks)
            if min_x < 36:
                issues.append("Margins appear too small")

        score = 10 - (len(issues) * 2)
        score = max(0, min(10, score))

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


def validate_technical_keywords(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])

        # normalize
        text = re.sub(r"\s+", " ", full_text).lower()

        found = []
        total_hits = 0

        for label in SRI_LANKAN_TECH_KEYWORDS:
            patterns = TECH_KEYWORD_VARIANTS.get(label, [])
            label_hit = 0

            for pat in patterns:
                matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
                if matches:
                    label_hit = len(matches)
                    break

            if label_hit:
                found.append(label)
                total_hits += label_hit

        found = list(dict.fromkeys(found))
        keyword_count = len(found)

        if keyword_count >= 10:
            status, message, score = "success", f"Excellent! Found {keyword_count} relevant technical keywords.", 10
        elif keyword_count >= 5:
            status, message, score = "success", f"Good! Found {keyword_count} technical keywords. Consider adding more.", 7
        elif keyword_count >= 3:
            status, message, score = "warning", f"Basic technical keywords found ({keyword_count}). Add more tools/frameworks.", 5
        else:
            status, message, score = "warning", f"Only {keyword_count} technical keywords found. Add a clear skills section.", 3

        return {
            "status": status,
            "message": message,
            "value": f"{keyword_count} keywords",
            "found_keywords": found[:12],
            "score": score,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


# -----------------------------
# GitHub link functions (safer)
# -----------------------------
def normalize_github_url(url: str) -> str:
    u = url.strip()
    if not u.lower().startswith("http"):
        u = "https://" + u
    u = u.split("#")[0].split("?")[0]
    return u.rstrip("/")


def is_github_repo_link(url: str) -> bool:
    u = url.split("?")[0].rstrip("/")
    parts = u.split("/")
    if len(parts) < 5:
        return False
    owner = parts[3]
    repo = parts[4]
    if not owner or not repo:
        return False
    bad = {"followers", "following", "repositories", "repos", "stars", "starred", "packages", "projects", "settings"}
    return repo.lower() not in bad


def github_url_exists(url: str):
    try:
        u = normalize_github_url(url)
        parsed = urlparse(u)

        host = (parsed.hostname or "").lower()
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


def extract_github_links_from_text(text: str):
    if not text:
        return []
    fixed = re.sub(r"(github\.com/)\s+", r"\1", text, flags=re.IGNORECASE)
    fixed = re.sub(r"(github\.com/[A-Za-z0-9_.-]+)\s+([A-Za-z0-9_.-]+)", r"\1\2", fixed, flags=re.IGNORECASE)
    pattern = r"(?:https?://)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?"
    urls = re.findall(pattern, fixed, flags=re.IGNORECASE)
    out = []
    for u in urls:
        if not u.lower().startswith("http"):
            u = "https://" + u
        out.append(u)
    return list(dict.fromkeys(out))


def validate_github_links(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        links = []
        pdf_links = bundle["links"] if bundle else []
        links.extend([u for u in pdf_links if "github.com" in u.lower()])

        if not links:
            md = get_markdown_cached(pdf_path, bundle) if bundle else pymupdf4llm.to_markdown(pdf_path)
            links = extract_github_links_from_text(md)

        links = [normalize_github_url(u) for u in links]
        links = list(dict.fromkeys(links))

        if not links:
            return {"status": "warning", "message": "No GitHub links found.", "repos": [], "value": "No links", "score": 3}

        repo_links = [u for u in links if is_github_repo_link(u)]
        profile_links = [u for u in links if u not in repo_links]

        details = []
        valid_profile = 0
        valid_repo = 0

        for u in profile_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "profile", "valid": ok, "message": msg})
            if ok:
                valid_profile += 1
            time.sleep(0.10)

        for u in repo_links:
            ok, msg = github_url_exists(u)
            details.append({"url": u, "type": "repo", "valid": ok, "message": msg})
            if ok:
                valid_repo += 1
            time.sleep(0.10)

        if valid_profile == 0 and valid_repo == 0:
            status = "warning"
            score = 3
        else:
            base = 6 if valid_profile > 0 else 0
            if valid_repo >= 3:
                boost = 4
            elif valid_repo == 2:
                boost = 3
            elif valid_repo == 1:
                boost = 2
            else:
                boost = 0
            score = min(10, base + boost)
            status = "success" if score >= 7 else "warning"

        return {
            "status": status,
            "message": f"GitHub found: {len(profile_links)} profile/org link(s), {len(repo_links)} repo link(s). "
            f"Working: {valid_profile} profile, {valid_repo} repo.",
            "repos": details,
            "value": f"{valid_profile} profile, {valid_repo} repo working",
            "score": score,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "repos": [], "value": "Error", "score": 0}


def find_specialization(pdf_path: str, bundle: Dict[str, Any] = None):
    specializations = {
        "Software Technology": "software technology",
        "Network Technology": "network technology",
        "Multimedia Technology": "multimedia technology",
    }
    try:
        text = (bundle["text_lower"] if bundle else "\n".join([p.get_text().lower() for p in pymupdf.open(pdf_path)]))
        text = text[:12000]  # early section usually enough
        for spec_name, keyword in specializations.items():
            if keyword in text:
                return {"status": "success", "message": f"Specialization: {spec_name}", "value": spec_name, "score": 10}
        return {"status": "warning", "message": "Specialization area is not clearly mentioned.", "value": "Not specified", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


import re
import pymupdf

# uses your existing global variables:
# - TECH_KEYWORD_VARIANTS (dict of {label: [regex patterns...]})
# - VALUED_SOFT_SKILLS (list of strings)
# - SRI_LANKAN_TECH_KEYWORDS (list of labels)

def check_skills_separation(pdf_path: str):
    """
    Detect:
      1) Separate 'Technical Skills' and 'Soft Skills' sections (best case)
      2) Single 'Skills' section (common) and auto-split into technical vs soft
    Returns counts + status + message + score.
    """

    # --- helpers ---
    def is_header(line_lower: str) -> bool:
        # common headers
        headers = [
            "skills", "skill set", "skill-set", "skill summary",
            "technical skills", "tech skills", "technical skill",
            "soft skills", "soft skill", "personal skills", "interpersonal skills",
            "technologies", "tech stack", "tools & technologies", "programming"
        ]
        return any(h == line_lower or line_lower.startswith(h) for h in headers)

    def split_items(line: str) -> list[str]:
        """
        Split a line into skill tokens:
        - bullets, commas, pipes, slashes
        """
        s = (line or "").strip()
        if not s:
            return []

        # remove leading bullets/symbols
        s = re.sub(r"^[•\-\*\u2022]+\s*", "", s)

        # split by common separators
        parts = re.split(r"[,\|\u2022/;]+", s)
        items = []
        for p in parts:
            p = p.strip()
            if len(p) < 2:
                continue
            # avoid long sentences
            if len(p) > 40:
                continue
            items.append(p)
        return items

    def is_tech_skill(token: str) -> bool:
        t = token.strip()
        if not t:
            return False
        low = t.lower()

        # match against your TECH_KEYWORD_VARIANTS
        for label, patterns in TECH_KEYWORD_VARIANTS.items():
            for pat in patterns:
                if re.search(pat, low, flags=re.IGNORECASE):
                    return True
        return False

    def is_soft_skill(token: str) -> bool:
        low = token.strip().lower()
        return any(s in low for s in VALUED_SOFT_SKILLS)

    stop_words = [
        "project", "projects", "experience", "education", "qualification",
        "reference", "references", "contact", "certification", "certificate",
        "achievement", "achievements", "profile", "summary", "declaration",
        "language", "languages", "interest", "interests", "volunteer"
    ]

    # --- main parse ---
    soft_skills = []
    technical_skills = []

    # section modes: "soft", "technical", "mixed", None
    current_section = None
    seen_any_skills_header = False
    mixed_section_lines = []

    soft_headers = ["soft skill", "soft skills", "personal skills", "interpersonal"]
    tech_headers = ["technical skill", "technical skills", "tech stack", "technologies", "programming", "tools"]

    try:
        with pymupdf.open(pdf_path) as doc:
            # collect blocks in reading order
            all_blocks = []
            for page in doc:
                for b in page.get_text("blocks"):
                    all_blocks.append(b)

        all_blocks.sort(key=lambda b: (b[1], b[0]))

        sorted_lines = []
        for b in all_blocks:
            text = b[4] or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    sorted_lines.append(line)

        for line in sorted_lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()

            if not line_clean:
                continue

            # stop header detection (exit skills area)
            is_stop_header = any(w in line_lower for w in stop_words) and len(line_clean) < 30
            if is_stop_header and "skill" not in line_lower:
                current_section = None
                continue

            # header detection
            if len(line_clean) < 35 and is_header(line_lower):
                seen_any_skills_header = True

                if any(h in line_lower for h in soft_headers):
                    current_section = "soft"
                elif any(h in line_lower for h in tech_headers):
                    current_section = "technical"
                else:
                    # "Skills" generic header
                    current_section = "mixed"
                continue

            # collect items depending on section
            if current_section == "soft":
                soft_skills.extend(split_items(line_clean))

            elif current_section == "technical":
                technical_skills.extend(split_items(line_clean))

            elif current_section == "mixed":
                # store lines first; we will auto-split later
                mixed_section_lines.append(line_clean)

        # If only mixed skills exists, auto-split using dictionaries
        if (not technical_skills and not soft_skills) and mixed_section_lines:
            tokens = []
            for l in mixed_section_lines:
                tokens.extend(split_items(l))

            for tok in tokens:
                if is_tech_skill(tok):
                    technical_skills.append(tok)
                elif is_soft_skill(tok):
                    soft_skills.append(tok)
                else:
                    # unknown token: ignore (or you can keep in "other")
                    pass

        # de-dup
        soft_skills = list(dict.fromkeys([s.strip() for s in soft_skills if s.strip()]))
        technical_skills = list(dict.fromkeys([s.strip() for s in technical_skills if s.strip()]))

        has_soft = len(soft_skills) > 0
        has_tech = len(technical_skills) > 0

        # scoring logic
        if has_soft and has_tech:
            return {
                "status": "success",
                "message": "Technical and Soft skills detected.",
                "value": "Both present ✓",
                "score": 10,
                "soft_count": len(soft_skills),
                "tech_count": len(technical_skills),
                "soft_samples": soft_skills[:8],
                "tech_samples": technical_skills[:12],
            }

        if has_tech and not has_soft:
            return {
                "status": "warning",
                "message": "Technical skills detected, but Soft Skills are not clear. Add a small Soft Skills list.",
                "value": "Technical only",
                "score": 7,
                "soft_count": 0,
                "tech_count": len(technical_skills),
                "tech_samples": technical_skills[:12],
            }

        if has_soft and not has_tech:
            return {
                "status": "warning",
                "message": "Soft skills detected, but Technical Skills are not clear. Add a clear Tech Stack/Skills list.",
                "value": "Soft only",
                "score": 6,
                "soft_count": len(soft_skills),
                "tech_count": 0,
                "soft_samples": soft_skills[:8],
            }

        # if we saw a header but couldn't parse items (PDF layout issues)
        if seen_any_skills_header:
            return {
                "status": "warning",
                "message": "Skills heading found, but individual skills are not readable (PDF formatting/layout issue). .",
                "value": "Unclear",
                "score": 4,
                "soft_count": 0,
                "tech_count": 0,
            }

        return {
            "status": "error",
            "message": "No clear Skills section detected. Add a Skills section with bullet points.",
            "value": "Not found",
            "score": 2,
            "soft_count": 0,
            "tech_count": 0,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}

def check_contact_information(pdf_path: str, bundle: Dict[str, Any] = None):
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text_lower = full_text.lower()

        phone_pattern = r"(\+94|0)?[\s-]?[0-9]{9,10}"
        has_phone = re.search(phone_pattern, full_text)

        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        has_email = re.search(email_pattern, full_text)

        has_linkedin = "linkedin" in text_lower
        has_location = any(word in text_lower for word in ["colombo", "sri lanka", "address", "location"])

        score = 0
        found = []
        missing = []

        if has_phone:
            score += 3
            found.append("Phone")
        else:
            missing.append("Phone")

        if has_email:
            score += 3
            found.append("Email")
        else:
            missing.append("Email")

        if has_linkedin:
            score += 2
            found.append("LinkedIn")
        else:
            missing.append("LinkedIn")

        if has_location:
            score += 2
            found.append("Location")
        else:
            missing.append("Location")

        if score >= 8:
            return {
                "status": "success",
                "message": f"Complete contact information found: {', '.join(found)}",
                "value": "Complete ✓",
                "score": 10,
                "details": found,
            }
        elif score >= 6:
            return {
                "status": "success",
                "message": f"Good contact info. Found: {', '.join(found)}",
                "value": "Good",
                "score": 8,
                "details": found,
            }
        else:
            return {
                "status": "warning",
                "message": f"Missing: {', '.join(missing)}. Add complete contact information.",
                "value": "Incomplete",
                "score": max(4, score),
                "details": found,
            }

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_action_verbs(pdf_path: str, bundle: Dict[str, Any] = None):
    strong_verbs = [
        "developed",
        "designed",
        "implemented",
        "created",
        "built",
        "led",
        "managed",
        "optimized",
        "improved",
        "analyzed",
        "achieved",
        "delivered",
        "collaborated",
        "architected",
        "engineered",
        "deployed",
        "integrated",
        "automated",
        "streamlined",
    ]
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])
        text_lower = full_text.lower()

        found_verbs = [v for v in strong_verbs if v in text_lower]
        verb_count = len(found_verbs)

        if verb_count >= 8:
            return {
                "status": "success",
                "message": f"Excellent! Found {verb_count} strong action verbs.",
                "value": f"{verb_count} verbs ✓",
                "score": 10,
                "verbs": found_verbs[:10],
            }
        elif verb_count >= 5:
            return {
                "status": "success",
                "message": f"Good use of action verbs ({verb_count} found).",
                "value": f"{verb_count} verbs",
                "score": 7,
                "verbs": found_verbs[:10],
            }
        elif verb_count >= 3:
            return {
                "status": "warning",
                "message": f"Only {verb_count} action verbs found. Use more impact words.",
                "value": f"{verb_count} verbs",
                "score": 5,
                "verbs": found_verbs[:10],
            }
        else:
            return {
                "status": "warning",
                "message": "Very few action verbs. Use words like 'developed', 'implemented', 'designed'.",
                "value": "Weak",
                "score": 3,
                "verbs": found_verbs[:10],
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_quantifiable_achievements(pdf_path: str, bundle: Dict[str, Any] = None):
    """
    FIXED:
    - Uses finditer (not findall with groups)
    - Counts real metric matches
    """
    try:
        full_text = bundle["text"] if bundle else "\n".join([p.get_text() for p in pymupdf.open(pdf_path)])

        number_patterns = [
            r"\b\d+%\b",  # 40%
            r"\b\d+\+\b",  # 10+
            r"\b\d+\s*(projects?|users?|customers?|members?|students?)\b",
            r"\b(?:increased|decreased|improved|reduced|grew)\b[^\n]{0,60}\b\d+\b",
            r"\b\d+\s*(years?|months?)\b",
        ]

        hits = []
        for pat in number_patterns:
            for m in re.finditer(pat, full_text, flags=re.IGNORECASE):
                hits.append(m.group(0))

        # de-duplicate for fair counting
        hits = list(dict.fromkeys([h.strip() for h in hits if h and h.strip()]))
        metric_count = len(hits)

        if metric_count >= 5:
            return {
                "status": "success",
                "message": f"Excellent! Found {metric_count} quantifiable achievements/metrics.",
                "value": f"{metric_count} metrics ✓",
                "score": 10,
                "examples": hits[:5],
            }
        elif metric_count >= 3:
            return {
                "status": "success",
                "message": f"Good quantification with {metric_count} metrics.",
                "value": f"{metric_count} metrics",
                "score": 7,
                "examples": hits[:5],
            }
        elif metric_count >= 1:
            return {
                "status": "warning",
                "message": f"Only {metric_count} metrics found. Add more numbers to show impact.",
                "value": f"{metric_count} metrics",
                "score": 5,
                "examples": hits[:5],
            }
        else:
            return {
                "status": "warning",
                "message": "No quantifiable achievements. Add numbers (e.g., '40% improvement', '5 projects').",
                "value": "None found",
                "score": 2,
                "examples": [],
            }

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}


def check_professional_summary(pdf_path: str, bundle: Dict[str, Any] = None):
    """
    Still checks top area (layout-based), but uses full doc open.
    """
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        first_page = doc[0]
        text = first_page.get_text().lower()

        summary_keywords = [
            "summary",
            "profile",
            "objective",
            "career objective",
            "professional summary",
            "about me",
            "introduction",
        ]

        has_summary = any(k in text for k in summary_keywords)

        blocks = first_page.get_text("blocks")
        if blocks:
            page_height = first_page.rect.height
            top_section = page_height * 0.3

            has_top_summary = False
            for block in blocks:
                if block[1] < top_section:
                    block_text = (block[4] or "").lower()
                    if any(k in block_text for k in summary_keywords):
                        has_top_summary = True
                        break

            if has_top_summary:
                return {"status": "success", "message": "Professional summary found at top of CV.", "value": "Present ✓", "score": 10}
            elif has_summary:
                return {"status": "success", "message": "Summary section found.", "value": "Present", "score": 8}
            else:
                return {"status": "info", "message": "No professional summary. Consider adding a brief career objective.", "value": "Not found", "score": 6}

        return {"status": "warning", "message": "Could not detect professional summary.", "value": "Not detected", "score": 5}

    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc:
            doc.close()


# -----------------------------
# Blind mode
# -----------------------------
def apply_blind_mode(results: Dict[str, Any], blind_mode: bool) -> Dict[str, Any]:
    if not blind_mode:
        return results
    adjusted = dict(results)

    if "photo" in adjusted:
        adjusted["photo"] = dict(adjusted["photo"])
        adjusted["photo"]["message"] = "Blind mode: photo ignored for fairness."
        adjusted["photo"]["score"] = 10
        adjusted["photo"]["status"] = "info"

    if "ol_al" in adjusted:
        adjusted["ol_al"] = dict(adjusted["ol_al"])
        adjusted["ol_al"]["message"] = "Blind mode: school results treated as low importance."
        adjusted["ol_al"]["score"] = max(adjusted["ol_al"].get("score", 0), 7)

    return adjusted


# -----------------------------
# Safe JSON parse 
# -----------------------------
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
                return {}
    return {}


# -----------------------------
# LLM validation 
# -----------------------------
def validate_with_llm(pdf_path: str, validation_results: Dict[str, Any], bundle: Dict[str, Any], temperature: float = 0.1):
    try:
        markdown_text = get_markdown_cached(pdf_path, bundle)

        groq_api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            return {"status": "error", "message": "API key missing.", "results": [], "passed": 0, "total": 9, "score": 0}

        
        context = f"""
VALIDATION CONTEXT (from automated checks):
- Photo detected: {'YES' if validation_results['photo']['status'] == 'success' else 'NO'}
- Email found: {'YES' if validation_results['professional_email']['status'] != 'error' else 'NO'}
- Email status: {validation_results['professional_email']['value']}
- Page count: {validation_results['page_count']['value']}
- O/L & A/L present: {'YES' if validation_results['ol_al']['status'] == 'warning' else 'NO'}
- Skills separation: {validation_results['skills']['value']}
- GitHub links: {validation_results['github']['value']}
- Technical keywords found: {validation_results['keywords']['value']}
"""

       
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

{context}

CV TEXT:
{markdown_text}
"""

        client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )

        raw = (response.choices[0].message.content or "").strip()
        data = safe_json_loads(raw)

        answers = data.get("answers", [])
        total = 9

        passed = 0
        for a in answers[:total]:
            if a.get("yes") is True:
                passed += 1

        score = round((passed / total) * 10, 1)

        return {"status": "success", "message": "AI done", "results": answers[:total], "passed": passed, "total": total, "score": score}

    except Exception as e:
        return {"status": "error", "message": f"AI Analysis error: {str(e)}", "results": [], "passed": 0, "total": 9, "score": 0}


# -----------------------------
# Overall score
# -----------------------------
def calculate_overall_score(results: Dict[str, Any], llm_std: float = 0.0):
    weights = {
        "page_count": 0.05,
        "gpa": 0.05,
        "professional_email": 0.08,
        "photo": 0.04,
        "ol_al": 0.05,
        "formatting": 0.06,
        "specialization": 0.04,
        "github": 0.08,
        "skills": 0.08,
        "keywords": 0.10,
        "contact": 0.06,
        "action_verbs": 0.07,
        "achievements": 0.07,
        "summary": 0.05,
        "llm": 0.10,
    }
   
    if llm_std >= 2.0:
        weights["llm"] = 0.05

    score_map = {}
    for k in weights.keys():
        if k == "llm":
            score_map[k] = results.get("llm", {}).get("score", 0)
        else:
            score_map[k] = results.get(k, {}).get("score", 0)

    total_score = 0.0
    total_weight = 0.0
    for key, w in weights.items():
        total_score += float(score_map.get(key, 0)) * float(w)
        total_weight += float(w)

    final_score = (total_score / total_weight) if total_weight else 0.0

    if final_score >= 8.5:
        grade, status = "Excellent", "success"
    elif final_score >= 7.0:
        grade, status = "Good", "success"
    elif final_score >= 5.5:
        grade, status = "Fair", "warning"
    else:
        grade, status = "Needs Improvement", "warning"

    return {
        "status": status,
        "score": round(final_score, 1),
        "grade": grade,
        "message": f"Overall CV Score: {round(final_score, 1)}/10 - {grade}",
        "llm_weight_adjusted": (llm_std >= 2.0),
    }


# -----------------------------
# Dimension scoring
# -----------------------------
def _get_score10(results: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = results.get(key, {})
        if isinstance(v, dict):
            s = v.get("score", default)
            return float(s) if s is not None else float(default)
        return float(default)
    except Exception:
        return float(default)


def _weighted_avg_10(items: List[tuple]) -> float:
    total = 0.0
    wsum = 0.0
    for s, w in items:
        total += float(s) * float(w)
        wsum += float(w)
    return (total / wsum) if wsum else 0.0


def _to_100(score10: float) -> float:
    return max(0.0, min(100.0, float(score10) * 10.0))


def calculate_dimension_scores(results: Dict[str, Any]) -> Dict[str, Any]:
    impact_10 = _weighted_avg_10(
        [
            (_get_score10(results, "achievements", 0), 0.45),
            (_get_score10(results, "action_verbs", 0), 0.35),
            (_get_score10(results, "summary", 0), 0.20),
        ]
    )

    brevity_10 = _weighted_avg_10([(_get_score10(results, "page_count", 0), 0.75), (_get_score10(results, "ol_al", 0), 0.25)])

    style_10 = _weighted_avg_10(
        [
            (_get_score10(results, "formatting", 0), 0.70),
            (_get_score10(results, "professional_email", 0), 0.20),
            (_get_score10(results, "photo", 0), 0.10),
        ]
    )

    skills_10 = _weighted_avg_10(
        [
            (_get_score10(results, "skills", 0), 0.40),
            (_get_score10(results, "keywords", 0), 0.40),
            (_get_score10(results, "github", 0), 0.20),
        ]
    )

    return {
        "impact": int(round(_to_100(impact_10), 0)),
        "brevity": int(round(_to_100(brevity_10), 0)),
        "style": int(round(_to_100(style_10), 0)),
        "skills": int(round(_to_100(skills_10), 0)),
        "debug": {
            "impact_10": round(impact_10, 2),
            "brevity_10": round(brevity_10, 2),
            "style_10": round(style_10, 2),
            "skills_10": round(skills_10, 2),
        },
    }


# -----------------------------
# Greeting
# -----------------------------
def get_time_greeting(now=None):
    now = now or datetime.now()
    h = now.hour
    if 5 <= h < 12:
        return "Good morning."
    elif 12 <= h < 17:
        return "Good afternoon."
    elif 17 <= h < 22:
        return "Good evening."
    else:
        return "Good night."


def fetch_stats(days: int = 30) -> Dict[str, Any]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # total runs
        cur.execute(
            "SELECT COUNT(*) FROM resume_runs WHERE created_at >= ?",
            (cutoff,),
        )
        total = cur.fetchone()[0] or 0

        # average overall score ONLY
        cur.execute(
            "SELECT AVG(overall_score) FROM resume_runs WHERE created_at >= ?",
            (cutoff,),
        )
        avg_overall = cur.fetchone()[0]
        avg_overall = round(avg_overall or 0, 2)

        # grade distribution
        cur.execute(
            """
            SELECT grade, COUNT(*)
            FROM resume_runs
            WHERE created_at >= ?
            GROUP BY grade
            ORDER BY COUNT(*) DESC
            """,
            (cutoff,),
        )
        grade_dist = [{"grade": g, "count": c} for (g, c) in cur.fetchall()]

        # last 20 runs
        cur.execute(
            """
            SELECT filename, overall_score, grade, llm_score, created_at
            FROM resume_runs
            WHERE created_at >= ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (cutoff,),
        )
        recent = [
            {
                "filename": f,
                "overall_score": s,
                "grade": gr,
                "llm_score": ls,
                "created_at": ca,
            }
            for (f, s, gr, ls, ca) in cur.fetchall()
        ]

        return {
            "days": days,
            "total_runs": total,
            "avg_overall": avg_overall,
            "grade_distribution": grade_dist,
            "recent_runs": recent,
        }

    finally:
        conn.close()


# -----------------------------
# Routes
# -----------------------------
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

        filename = secure_filename(file.filename)
        temp_name = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
        file.save(filepath)

        results = None
        recommendations = []
        greeting = get_time_greeting()
        sub_greeting = "Welcome to your resume review."

        # Create preview BEFORE deleting temp file
        preview_info = {"preview_id": "", "preview_filename": ""}
        try:
            preview_info = save_preview_file(filepath, filename)

            file_hash = sha256_file(filepath)
            bundle = extract_pdf_bundle(filepath)

            results = {
                "page_count": check_cv_page_count(filepath, bundle),
                "gpa": check_gpa_in_cv(filepath, bundle),
                "professional_email": check_professional_email(filepath, bundle),
                "photo": check_photo_presence(filepath, bundle),
                "ol_al": check_ol_al_presence(filepath, bundle),
                "formatting": check_formatting_quality(filepath),
                "specialization": find_specialization(filepath, bundle),
                "github": validate_github_links(filepath, bundle),
                "skills": check_skills_separation(filepath),
                "keywords": validate_technical_keywords(filepath, bundle),
                "contact": check_contact_information(filepath, bundle),
                "action_verbs": check_action_verbs(filepath, bundle),
                "achievements": check_quantifiable_achievements(filepath, bundle),
                "summary": check_professional_summary(filepath, bundle),
            }

            results["llm"] = validate_with_llm(filepath, results, bundle)
            results = apply_blind_mode(results, blind_mode=blind_mode)

            results["overall"] = calculate_overall_score(results)
            results["dimensions"] = calculate_dimension_scores(results)

            # Get job recommendations based on CV keywords
            cv_keywords = results['keywords'].get('found_keywords', [])

        # IMPORTANT: pass cv_text so we can avoid senior roles for junior CVs
            results["job_recommendations"] = get_job_recommendations(
            results=results,
            cv_text=bundle.get("text", ""),
            location_filter=request.form.get("location_filter", "All"),
            top_n=5,
            seniority_mode="filter"
            )


            # Provide preview URL to template instead of base64
            results["preview_id"] = preview_info["preview_id"]
            results["preview_filename"] = preview_info["preview_filename"]
            results["preview_url"] = url_for(
                "preview_file", preview_id=preview_info["preview_id"], filename=preview_info["preview_filename"]
            )

            try:
                save_resume_run(
                    file_hash=file_hash,
                    filename=filename,
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
            recommendations=recommendations,
            filename=filename,
            greeting=greeting,
            sub_greeting=sub_greeting,
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

    filename = secure_filename(file.filename)
    temp_name = f"{int(time.time())}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
    file.save(filepath)

    preview_info = {"preview_id": "", "preview_filename": ""}
    try:
        preview_info = save_preview_file(filepath, filename)

        bundle = extract_pdf_bundle(filepath)
        results = {
            "page_count": check_cv_page_count(filepath, bundle),
            "gpa": check_gpa_in_cv(filepath, bundle),
            "professional_email": check_professional_email(filepath, bundle),
            "photo": check_photo_presence(filepath, bundle),
            "ol_al": check_ol_al_presence(filepath, bundle),
            "formatting": check_formatting_quality(filepath),
            "specialization": find_specialization(filepath, bundle),
            "github": validate_github_links(filepath, bundle),
            "skills": check_skills_separation(filepath),
            "keywords": validate_technical_keywords(filepath, bundle),
            "contact": check_contact_information(filepath, bundle),
            "action_verbs": check_action_verbs(filepath, bundle),
            "achievements": check_quantifiable_achievements(filepath, bundle),
            "summary": check_professional_summary(filepath, bundle),
        }

        results["llm"] = validate_with_llm(filepath, results, bundle)
        results = apply_blind_mode(results, blind_mode=blind_mode)
        results["overall"] = calculate_overall_score(results)
        results["dimensions"] = calculate_dimension_scores(results)

       
        cv_keywords = results["keywords"].get("found_keywords", [])

        results["job_recommendations"] = get_job_recommendations(
        results=results,
        cv_text=bundle.get("text", ""),
        location_filter=request.form.get("location_filter", "All"),
        top_n=5,
        seniority_mode="filter"
        )


        results["preview_id"] = preview_info["preview_id"]
        results["preview_filename"] = preview_info["preview_filename"]
        results["preview_url"] = url_for("preview_file", preview_id=preview_info["preview_id"], filename=preview_info["preview_filename"])

        return jsonify(results)
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@app.route("/stats")
def stats():
    stats_data = fetch_stats()
    return render_template("stats.html", stats=stats_data)


@app.route("/job-recommendations")
def job_recommendations():
    return render_template("job_recommendation.html")


@app.route("/preview/<preview_id>/<filename>")
def preview_file(preview_id, filename):
    preview_id = re.sub(r"[^a-f0-9]", "", preview_id.lower())  # uuid hex only
    filename = secure_filename(filename)

    user_preview_dir = os.path.join(PREVIEW_FOLDER, preview_id)
    file_path = os.path.join(user_preview_dir, filename)

    if not preview_id or not filename or not os.path.exists(file_path):
        abort(404)

    return send_from_directory(user_preview_dir, filename, mimetype="application/pdf")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Enhanced CV Validator for Sri Lankan IT Market")
    print("  With Job Recommendations & Statistics")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)