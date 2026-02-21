from flask import Flask, request, render_template, jsonify, session
import os
import pymupdf
import pymupdf4llm
import re
import requests
import time
import gc
from werkzeug.utils import secure_filename
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get API key from config file
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = None

app = Flask(__name__)
app.secret_key = "cv_validator_secret_key_2025"   # session use කරන්න
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============= Configuration =============

SRI_LANKAN_TECH_KEYWORDS = [
    "Java", "Python", "JavaScript", "C#", "C++",
    "MySQL", "PostgreSQL", "MongoDB", "Oracle",
    "Git", "GitHub", "GitLab",
    "HTML5", "CSS3", "React", "Angular", "Vue.js",
    "Node.js", "Express", "Django", "Flask", "Spring Boot",
    "Android", "iOS", "Flutter", "React Native",
    "AWS", "Azure", "Docker", "Kubernetes",
    "REST API", "GraphQL", "Microservices",
    "Agile", "Scrum", "JIRA"
]

VALUED_SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "adaptability", "collaboration",
    "creativity", "attention to detail", "work ethic", "interpersonal"
]

# ============= Job Recommendation Functions =============

def load_job_data():
    """Load job data from CSV file"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'topjobs_it_jobs.csv')
        df = pd.read_csv(csv_path)
        df = df.dropna(how='all')
        df['title']       = df['title'].fillna('Not specified')
        df['company']     = df['company'].fillna('Company Name Withheld')
        df['location']    = df['location'].fillna('Sri Lanka')
        df['description'] = df['description'].fillna('Please refer the vacancy')
        df['url']         = df['url'].fillna('')
        df['closing_date']= df.get('closing_date', pd.Series([''] * len(df))).fillna('')
        df['combined_text'] = df['title'].astype(str) + ' ' + df['description'].astype(str)
        # Remove completely empty / placeholder rows
        df = df[df['title'] != 'Not specified'].reset_index(drop=True)
        # Remove duplicates by job_number if column exists
        if 'job_number' in df.columns:
            df = df.drop_duplicates(subset=['job_number']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading job data: {e}")
        return None


def get_job_recommendations(cv_keywords, location_filter='All', min_match_percentage=0):
    """Return all jobs ranked by TF-IDF cosine similarity against CV keywords."""
    try:
        df = load_job_data()
        if df is None or len(df) == 0:
            return []

        if location_filter and location_filter != 'All':
            df = df[df['location'].str.contains(location_filter, case=False, na=False)]

        if len(df) == 0:
            return []

        user_skills = ' '.join(cv_keywords) if cv_keywords else 'software developer intern'

        all_text   = list(df['combined_text']) + [user_skills]
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        tfidf_matrix  = vectorizer.fit_transform(all_text)

        user_vector  = tfidf_matrix[-1]
        job_vectors  = tfidf_matrix[:-1]
        similarities = cosine_similarity(user_vector, job_vectors).flatten()

        df = df.copy()
        df['similarity_score'] = similarities
        df = df.sort_values('similarity_score', ascending=False)

        if min_match_percentage > 0:
            df = df[df['similarity_score'] * 100 >= min_match_percentage]

        jobs = []
        for _, row in df.iterrows():
            mp = row['similarity_score'] * 100
            # Determine matched / missing skills (simple keyword overlap)
            title_desc = (str(row['title']) + ' ' + str(row['description'])).lower()
            matched  = [k for k in cv_keywords if k.lower() in title_desc]
            # cv_level from overall score will be enriched later; placeholder here
            jobs.append({
                'title':           row['title'],
                'company':         row['company'],
                'location':        row['location'],
                'description':     (str(row['description'])[:200] + '...'
                                    if len(str(row['description'])) > 200
                                    else str(row['description'])),
                'url':             row.get('url', ''),
                'closing_date':    str(row.get('closing_date', '')),
                'match_percentage': round(mp, 1),
                'match_level':     ('Excellent' if mp >= 70 else 'Good' if mp >= 50 else 'Potential'),
                'matched_skills':  matched[:8],
                'missing_skills':  [],          # can extend later
                'cv_level_used':   'Keywords',
            })
        return jobs

    except Exception as e:
        print(f"Error in job recommendations: {e}")
        return []

# ============= CV Validation Functions =============

def check_cv_page_count(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        if page_count == 1:
            return {"status": "success", "message": "Perfect! Single page CV.",      "value": "1 page ✓",  "score": 10}
        elif page_count == 2:
            return {"status": "success", "message": "Good! Two pages is acceptable.", "value": "2 pages ✓", "score": 8}
        else:
            return {"status": "warning", "message": f"CV has {page_count} pages. Keep it to 1-2 pages.", "value": f"{page_count} pages", "score": 4}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_gpa_in_cv(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        text_lower = full_text.lower()
        gpa_patterns = [r'gpa[:\s]+([0-9.]+)', r'cgpa[:\s]+([0-9.]+)', r'grade point[:\s]+([0-9.]+)']
        gpa_found = any(re.search(p, text_lower) for p in gpa_patterns)
        if gpa_found or 'gpa' in text_lower or 'cgpa' in text_lower:
            return {"status": "success", "message": "GPA is mentioned in the CV.", "value": "Found ✓", "score": 10}
        else:
            return {"status": "warning", "message": "GPA not mentioned. If you have a good GPA (>3.0), add it.", "value": "Not found", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_professional_email(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text)
        if not emails:
            return {"status": "error", "message": "No email address found in CV.", "value": "Missing", "score": 0}
        bad_words = ['cool','hot','sexy','prince','king','boss','gangsta','swag','legend','hero','cutie','angel','devil','ninja']
        issues = []
        for email in emails:
            local = email.lower().split('@')[0]
            for w in bad_words:
                if w in local:
                    issues.append(f"{email} contains unprofessional word '{w}'")
            if re.search(r'\d{4,}', local):
                issues.append(f"{email} has too many consecutive numbers")
            if len(re.findall(r'[._-]', local)) > 2:
                issues.append(f"{email} has too many special characters")
        if issues:
            return {"status": "warning", "message": f"Email concerns: {'; '.join(issues)}", "value": "Needs improvement", "emails": emails, "score": 4}
        return {"status": "success", "message": "Email address looks professional.", "value": "Professional ✓", "emails": emails, "score": 10}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_photo_presence(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        has_image = any(doc[p].get_images(full=True) for p in range(len(doc)))
        if has_image:
            return {"status": "success", "message": "Professional photo found (good for Sri Lankan market).", "value": "Present ✓", "score": 10}
        return {"status": "info", "message": "No photo detected. Consider adding a professional photo.", "value": "Not found", "score": 7}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_ol_al_presence(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc).lower()
        has_ol = any(p in full_text for p in ['o/l','o.l','ordinary level','g.c.e o/l','gce o/l'])
        has_al = any(p in full_text for p in ['a/l','a.l','advanced level','g.c.e a/l','gce a/l'])
        if has_ol or has_al:
            details = (["O/L"] if has_ol else []) + (["A/L"] if has_al else [])
            return {"status": "warning",
                    "message": f"School exam results ({', '.join(details)}) found. IT companies typically don't require these.",
                    "value": "Present (consider removing)", "score": 5}
        return {"status": "success", "message": "No school exam results. Good - focus on degree and technical skills.", "value": "Not present ✓", "score": 10}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_formatting_quality(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        font_sizes = set()
        issues = []
        for page_num in range(len(doc)):
            for block in doc[page_num].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.add(round(span["size"], 1))
        if len(font_sizes) < 2:   issues.append("Use different font sizes for headings and body text")
        elif len(font_sizes) > 6: issues.append("Too many different font sizes - keep it consistent")
        blocks = doc[0].get_text("blocks")
        if blocks and min(b[0] for b in blocks) < 36:
            issues.append("Margins appear too small")
        score = max(0, min(10, 10 - len(issues) * 2))
        if issues:
            return {"status": "warning", "message": f"Formatting suggestions: {'; '.join(issues)}", "value": f"{score}/10", "score": score}
        return {"status": "success", "message": "Formatting looks good.", "value": "Good ✓", "score": 10}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 5}
    finally:
        if doc: doc.close()


def validate_technical_keywords(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        found = [k for k in SRI_LANKAN_TECH_KEYWORDS
                 if re.search(r'\b' + re.escape(k) + r'\b', full_text, re.IGNORECASE)]
        n = len(found)
        if n >= 10: return {"status": "success", "message": f"Excellent! Found {n} relevant technical keywords.", "value": f"{n} keywords", "found_keywords": found, "score": 10}
        if n >= 5:  return {"status": "success", "message": f"Good! Found {n} technical keywords. Consider adding more.", "value": f"{n} keywords", "found_keywords": found, "score": 7}
        return {"status": "warning", "message": f"Only {n} technical keywords found. Add more relevant technologies.", "value": f"{n} keywords", "found_keywords": found, "score": 4}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "found_keywords": [], "score": 0}
    finally:
        if doc: doc.close()


def validate_github_links(pdf_path):
    try:
        markdown = pymupdf4llm.to_markdown(pdf_path)
        pattern  = r'https?://github\.com/[\w\-]+/[\w\-]+'
        repos    = list(set(re.findall(pattern, markdown)))
        if not repos:
            return {"status": "warning", "message": "No GitHub links found. Add GitHub projects to strengthen your CV.", "repos": [], "value": "No links", "score": 3}
        valid_count  = 0
        repo_details = []
        for repo in repos:
            try:
                r = requests.head(repo, timeout=5, allow_redirects=True)
                ok = r.status_code == 200
                msg = "Working" if ok else f"Error {r.status_code}"
            except:
                ok, msg = False, "Connection error"
            repo_details.append({"url": repo, "valid": ok, "message": msg})
            if ok: valid_count += 1
            time.sleep(0.3)
        status = "success" if valid_count == len(repos) else ("warning" if valid_count > 0 else "error")
        score  = 10 if valid_count == len(repos) else (6 if valid_count > 0 else 2)
        return {"status": status, "message": f"Found {len(repos)} links, {valid_count} are working.", "repos": repo_details, "value": f"{valid_count}/{len(repos)} working", "score": score}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "repos": [], "value": "Error", "score": 0}


def find_specialization(pdf_path):
    specs = {"Software Technology": "software technology", "Network Technology": "network technology", "Multimedia Technology": "multimedia technology"}
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        for pnum in range(min(3, len(doc))):
            text = doc[pnum].get_text().lower()
            for name, kw in specs.items():
                if kw in text:
                    return {"status": "success", "message": f"Specialization: {name}", "value": name, "score": 10}
        return {"status": "warning", "message": "Specialization area is not clearly mentioned.", "value": "Not specified", "score": 5}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_skills_separation(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        all_blocks = []
        for page in doc:
            all_blocks.extend(page.get_text("blocks"))
        all_blocks.sort(key=lambda b: (b[1], b[0]))
        lines = [line.strip() for b in all_blocks for line in b[4].splitlines()]

        soft_headers = ["soft skill","soft skills","interpersonal","personal skills"]
        tech_headers = ["technical skill","technical skills","tech stack","technologies","programming"]
        stop_words   = ["project","experience","education","qualification","reference","contact","certification","certificate","achievement","profile","summary","declaration","language","interest","volunteer"]

        current_section = None
        soft_skills, tech_skills = [], []

        for line in lines:
            ll = line.lower()
            if not line: continue
            if any(h in ll for h in soft_headers) and len(line) < 30:
                current_section = "soft"; continue
            if any(h in ll for h in tech_headers) and len(line) < 30:
                current_section = "technical"; continue
            if any(w in ll for w in stop_words) and len(line) < 30 and "skill" not in ll:
                current_section = None; continue
            if current_section == "soft" and len(line) > 2 and "skill" not in ll:
                soft_skills.append(line)
            elif current_section == "technical" and len(line) > 1 and "skill" not in ll:
                tech_skills.append(line)

        soft_skills = list(dict.fromkeys(soft_skills))
        tech_skills = list(dict.fromkeys(tech_skills))

        if soft_skills and tech_skills:
            return {"status": "success", "message": "Both Technical and Soft Skills sections found.", "value": "Both present ✓", "score": 10, "soft_count": len(soft_skills), "tech_count": len(tech_skills)}
        if tech_skills:
            return {"status": "warning", "message": "Technical skills found, but Soft Skills section missing.", "value": "Partial", "score": 6, "tech_count": len(tech_skills), "soft_count": 0}
        if soft_skills:
            return {"status": "warning", "message": "Soft Skills found, but Technical Skills section missing.", "value": "Partial", "score": 5, "soft_count": len(soft_skills), "tech_count": 0}
        return {"status": "error", "message": "Could not identify clear skill sections.", "value": "Not found", "score": 2, "soft_count": 0, "tech_count": 0}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error", "score": 0}
    finally:
        if doc: doc.close()


def check_contact_information(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        full_text  = "".join(page.get_text() for page in doc)
        text_lower = full_text.lower()
        has_phone    = bool(re.search(r'(\+94|0)?[\s-]?[0-9]{9,10}', full_text))
        has_email    = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text))
        has_linkedin = 'linkedin' in text_lower
        has_location = any(w in text_lower for w in ['colombo','sri lanka','address','location'])
        score, found, missing = 0, [], []
        for flag, pts, name in [(has_phone,3,"Phone"),(has_email,3,"Email"),(has_linkedin,2,"LinkedIn"),(has_location,2,"Location")]:
            if flag: score += pts; found.append(name)
            else: missing.append(name)
        if score >= 8: return {"status":"success","message":f"Complete contact info: {', '.join(found)}","value":"Complete ✓","score":10,"details":found}
        if score >= 6: return {"status":"success","message":f"Good contact info. Found: {', '.join(found)}","value":"Good","score":8,"details":found}
        return {"status":"warning","message":f"Missing: {', '.join(missing)}. Add complete contact info.","value":"Incomplete","score":max(4,score),"details":found}
    except Exception as e:
        return {"status":"error","message":f"Error: {e}","value":"Error","score":0}
    finally:
        if doc: doc.close()


def check_action_verbs(pdf_path):
    verbs = ['developed','designed','implemented','created','built','led','managed','optimized','improved','analyzed','achieved','delivered','collaborated','architected','engineered','deployed','integrated','automated','streamlined']
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        text  = "".join(page.get_text() for page in doc).lower()
        found = [v for v in verbs if v in text]
        n = len(found)
        if n >= 8: return {"status":"success","message":f"Excellent! Found {n} strong action verbs.","value":f"{n} verbs ✓","score":10,"verbs":found[:10]}
        if n >= 5: return {"status":"success","message":f"Good use of action verbs ({n} found).","value":f"{n} verbs","score":7,"verbs":found}
        if n >= 3: return {"status":"warning","message":f"Only {n} action verbs found. Use more impact words.","value":f"{n} verbs","score":5,"verbs":found}
        return {"status":"warning","message":"Very few action verbs. Use words like 'developed', 'implemented'.","value":"Weak","score":3,"verbs":found}
    except Exception as e:
        return {"status":"error","message":f"Error: {e}","value":"Error","score":0}
    finally:
        if doc: doc.close()


def check_quantifiable_achievements(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        patterns = [r'\d+%', r'\d+\+', r'\d+\s*(projects?|users?|customers?|members?|students?)',
                    r'(increased|decreased|improved|reduced|grew)\s+.*?\d+', r'\d+\s*(years?|months?)']
        metrics = [m for p in patterns for m in re.findall(p, text, re.IGNORECASE)]
        n = len(metrics)
        if n >= 5: return {"status":"success","message":f"Excellent! Found {n} quantifiable achievements.","value":f"{n} metrics ✓","score":10}
        if n >= 3: return {"status":"success","message":f"Good quantification with {n} metrics.","value":f"{n} metrics","score":7}
        if n >= 1: return {"status":"warning","message":f"Only {n} metrics found. Add more numbers to show impact.","value":f"{n} metrics","score":5}
        return {"status":"warning","message":"No quantifiable achievements. Add numbers (e.g., '40% improvement').","value":"None found","score":2}
    except Exception as e:
        return {"status":"error","message":f"Error: {e}","value":"Error","score":0}
    finally:
        if doc: doc.close()


def check_professional_summary(pdf_path):
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        first_page = doc[0]
        text = first_page.get_text().lower()
        kws  = ['summary','profile','objective','career objective','professional summary','about me','introduction']
        blocks     = first_page.get_text("blocks")
        top_cutoff = first_page.rect.height * 0.3
        for block in blocks:
            if block[1] < top_cutoff and any(k in block[4].lower() for k in kws):
                return {"status":"success","message":"Professional summary found at top of CV.","value":"Present ✓","score":10}
        if any(k in text for k in kws):
            return {"status":"success","message":"Summary section found.","value":"Present","score":8}
        return {"status":"info","message":"No professional summary. Consider adding a brief career objective.","value":"Not found","score":6}
    except Exception as e:
        return {"status":"error","message":f"Error: {e}","value":"Error","score":0}
    finally:
        if doc: doc.close()


def validate_with_llm(pdf_path, validation_results):
    try:
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        context = f"""
VALIDATION CONTEXT:
- Photo detected: {'YES' if validation_results['photo']['status'] == 'success' else 'NO'}
- Email status: {validation_results['professional_email']['value']}
- Page count: {validation_results['page_count']['value']}
- O/L & A/L present: {'YES' if validation_results['ol_al']['status'] == 'warning' else 'NO'}
- Skills separation: {validation_results['skills']['value']}
- GitHub links: {validation_results['github']['value']}
- Technical keywords: {validation_results['keywords']['value']}
"""
        query = f'''You are an HR manager at a leading Sri Lankan IT company.
Review this CV for an internship/junior developer position.

{context}

CV Content:
{markdown_text}

Answer YES or NO to each criterion (one per line):
1) Does it have a professional photo?
2) Is the email address professional?
3) Are technical skills clearly shown in a dedicated section?
4) Are projects described with technologies used?
5) Does it look clean, organized, and well-formatted?
6) Are school exam results (O/L, A/L) appropriately handled? (YES = NOT present)
7) Is it the right length (1-2 pages)?
8) Are required technical keywords present?
9) Is the ICT degree mentioned?
10) Are references present?'''

        groq_api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            return {"status":"error","message":"API key missing.","results":[],"passed":0,"total":10,"score":0}

        client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        t0 = time.time()
        response = client.chat.completions.create(
            messages=[{"role":"user","content":query}],
            model="llama-3.3-70b-versatile", temperature=0.1)
        elapsed = time.time() - t0
        answers = response.choices[0].message.content.strip().splitlines()

        criteria = [
            ("Professional Photo","Does it have a professional photo?"),
            ("Professional Email","Is the email address professional?"),
            ("Technical Skills Section","Are technical skills clearly shown?"),
            ("Project Technologies","Are technologies mentioned in projects?"),
            ("Clean Formatting","Is the CV well-formatted and organized?"),
            ("School Results Handling","Are O/L & A/L results appropriately handled?"),
            ("Appropriate Length","Is it 1-2 pages long?"),
            ("Technical Keywords","Are required technical keywords present?"),
            ("Degree Mentioned","Is the ICT degree mentioned?"),
            ("References Present","Are references included?"),
        ]
        results_list, passed = [], 0
        for i, ans in enumerate(answers[:len(criteria)]):
            is_yes = 'yes' in ans.strip().lower()
            results_list.append({"name":criteria[i][0],"description":criteria[i][1],"passed":is_yes})
            if is_yes: passed += 1

        return {"status":"success","message":f"AI Analysis complete ({elapsed:.1f}s)","results":results_list,"passed":passed,"total":len(criteria),"score":round(passed/len(criteria)*10,1)}

    except Exception as e:
        return {"status":"error","message":f"AI Analysis error: {str(e)}","results":[],"passed":0,"total":10,"score":0}


def calculate_overall_score(results):
    weights = {
        'page_count':0.05,'gpa':0.05,'professional_email':0.08,'photo':0.04,
        'ol_al':0.05,'formatting':0.06,'specialization':0.04,'github':0.08,
        'skills':0.08,'keywords':0.10,'contact':0.06,'action_verbs':0.07,
        'achievements':0.07,'summary':0.05,'llm':0.10,
    }
    total_score = sum(results.get(k,{}).get('score',0) * w for k,w in weights.items())
    final = round(total_score, 1)
    if final >= 8.5: grade, status = "Excellent", "success"
    elif final >= 7.0: grade, status = "Good", "success"
    elif final >= 5.5: grade, status = "Fair", "warning"
    else: grade, status = "Needs Improvement", "warning"
    return {"status":status,"score":final,"grade":grade,"message":f"Overall CV Score: {final}/10 - {grade}"}


def calculate_dimensions(results):
    """Calculate Impact / Brevity / Style / Skills dimension scores (0-100)"""
    impact  = round((results['action_verbs'].get('score',0)*0.4 + results['achievements'].get('score',0)*0.4 + results['summary'].get('score',0)*0.2)*10)
    brevity = round((results['page_count'].get('score',0)*0.6 + results['formatting'].get('score',0)*0.4)*10)
    style   = round((results['formatting'].get('score',0)*0.5 + results['professional_email'].get('score',0)*0.3 + results['contact'].get('score',0)*0.2)*10)
    skills  = round((results['keywords'].get('score',0)*0.5 + results['skills'].get('score',0)*0.3 + results['github'].get('score',0)*0.2)*10)
    return {"impact":impact,"brevity":brevity,"style":style,"skills":skills}


def get_greeting(score):
    if score >= 85: return "🚀 Outstanding!", "Your CV is highly competitive for Sri Lankan IT roles."
    if score >= 70: return "👍 Looking Good!", "A few tweaks will make your CV even stronger."
    if score >= 55: return "⚠️ Getting There!", "Several improvements needed to meet industry expectations."
    return "❗ Needs Work", "Follow the recommendations below to significantly improve your CV."

# ============= Routes =============

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'cv_file' not in request.files:
            return 'Please select a CV file.'
        file = request.files['cv_file']
        if file.filename == '' or not file.filename.endswith('.pdf'):
            return 'Please upload a PDF file only.'

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            results = {
                'page_count':       check_cv_page_count(filepath),
                'gpa':              check_gpa_in_cv(filepath),
                'professional_email': check_professional_email(filepath),
                'photo':            check_photo_presence(filepath),
                'ol_al':            check_ol_al_presence(filepath),
                'formatting':       check_formatting_quality(filepath),
                'specialization':   find_specialization(filepath),
                'github':           validate_github_links(filepath),
                'skills':           check_skills_separation(filepath),
                'keywords':         validate_technical_keywords(filepath),
                'contact':          check_contact_information(filepath),
                'action_verbs':     check_action_verbs(filepath),
                'achievements':     check_quantifiable_achievements(filepath),
                'summary':          check_professional_summary(filepath),
            }
            results['llm']      = validate_with_llm(filepath, results)
            results['overall']  = calculate_overall_score(results)
            results['dimensions'] = calculate_dimensions(results)

            # Job recommendations
            cv_keywords = results['keywords'].get('found_keywords', [])
            job_recs    = get_job_recommendations(cv_keywords)
            results['job_recommendations'] = job_recs

            # Greeting
            overall_100 = round(results['overall']['score'] * 10)
            greeting, sub_greeting = get_greeting(overall_100)

            # Save to session for job recommendations page
            session['last_results']  = results
            session['last_filename'] = filename

        finally:
            gc.collect()
            time.sleep(0.3)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except PermissionError:
                pass

        return render_template("index.html",
                               results=results,
                               filename=filename,
                               greeting=greeting,
                               sub_greeting=sub_greeting,
                               job_recommendations=job_recs)

    return render_template("index.html", results=None)


@app.route('/job-recommendations')
def job_recommendations():
    """Job recommendations page - reads from session or re-uses last results."""
    results  = session.get('last_results')
    filename = session.get('last_filename', '')

    if results:
        jobs         = results.get('job_recommendations', [])
        all_locations = sorted({j['location'] for j in jobs if j.get('location')})
        all_levels    = sorted({j['match_level'] for j in jobs if j.get('match_level')})
    else:
        jobs, all_locations, all_levels = [], [], []

    # Optional query-string filters
    loc_filter   = request.args.get('location', 'All')
    level_filter = request.args.get('level', 'All')

    filtered = jobs
    if loc_filter   != 'All': filtered = [j for j in filtered if j.get('location') == loc_filter]
    if level_filter != 'All': filtered = [j for j in filtered if j.get('match_level') == level_filter]

    return render_template("job_recommendation.html",
                           results=results,
                           filename=filename,
                           jobs=filtered,
                           all_locations=all_locations,
                           all_levels=all_levels,
                           loc_filter=loc_filter,
                           level_filter=level_filter)


@app.route('/stats')
def stats():
    """Statistics page - placeholder data."""
    stats_data = {
        "days": 30,
        "total_runs": 0,
        "avg_overall": "N/A",
        "grade_distribution": [],
        "recent_runs": [],
    }
    return render_template("stats.html", stats=stats_data)


@app.route('/api/validate', methods=['POST'])
def api_validate():
    if 'cv_file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['cv_file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files accepted"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        results = {
            'page_count': check_cv_page_count(filepath),
            'gpa': check_gpa_in_cv(filepath),
            'professional_email': check_professional_email(filepath),
            'photo': check_photo_presence(filepath),
            'ol_al': check_ol_al_presence(filepath),
            'formatting': check_formatting_quality(filepath),
            'specialization': find_specialization(filepath),
            'github': validate_github_links(filepath),
            'skills': check_skills_separation(filepath),
            'keywords': validate_technical_keywords(filepath),
            'contact': check_contact_information(filepath),
            'action_verbs': check_action_verbs(filepath),
            'achievements': check_quantifiable_achievements(filepath),
            'summary': check_professional_summary(filepath),
        }
        results['llm']     = validate_with_llm(filepath, results)
        results['overall'] = calculate_overall_score(results)
        cv_keywords = results['keywords'].get('found_keywords', [])
        results['job_recommendations'] = get_job_recommendations(cv_keywords)
        return jsonify(results)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  CV Validator Pro - Enhanced UI + Job Recommendations")
    print("  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)