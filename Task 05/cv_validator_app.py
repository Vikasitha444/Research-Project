from flask import Flask, request, render_template_string
import os
import pymupdf
import pymupdf4llm
import re
import requests
import time
from werkzeug.utils import secure_filename
from openai import OpenAI

# Config file එකෙන් API key ගන්න
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============= Validation Functions =============

def check_cv_page_count(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        doc.close()

        if page_count > 1:
            return {
                "status": "warning",
                "message": f"CV එකේ pages {page_count}ක් තියෙනවා. එක page එකක් විතරක් recommend කරනවා.",
                "value": f"{page_count} pages"
            }
        else:
            return {
                "status": "success",
                "message": "Page count හරි - එක page එකක් තියෙනවා.",
                "value": "1 page ✓"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error"}

def check_gpa_in_cv(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        text_lower = full_text.lower()

        if 'gpa' in text_lower or 'cgpa' in text_lower or 'grade point' in text_lower:
            return {
                "status": "success",
                "message": "GPA එක CV එකේ තියෙනවා.",
                "value": "Found ✓"
            }
        else:
            return {
                "status": "warning",
                "message": "GPA එක mention කරලා නෑ. හොඳ GPA එකක් තියෙනවා නම් add කරන්න.",
                "value": "Not found"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error"}

def extract_github_links(text):
    repo_pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
    repos = re.findall(repo_pattern, text)
    return list(set(repos))

def check_repository_exists(repo_url):
    try:
        response = requests.head(repo_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return True, "Working"
        elif response.status_code == 404:
            return False, "Not found"
        else:
            return False, f"Error {response.status_code}"
    except:
        return False, "Connection error"

def validate_github_links(pdf_path):
    try:
        markdown = pymupdf4llm.to_markdown(pdf_path)
        repos = extract_github_links(markdown)

        if not repos:
            return {
                "status": "warning",
                "message": "GitHub links නෑ. Projects තියෙනවා නම් GitHub links add කරන්න.",
                "repos": [],
                "value": "No links"
            }

        valid_count = 0
        repo_details = []

        for repo in repos:
            exists, message = check_repository_exists(repo)
            repo_details.append({"url": repo, "valid": exists, "message": message})
            if exists:
                valid_count += 1
            time.sleep(0.3)

        status = "success" if valid_count == len(repos) else "warning" if valid_count > 0 else "error"

        return {
            "status": status,
            "message": f"{len(repos)} links හම්බුණා, {valid_count} ක් working.",
            "repos": repo_details,
            "value": f"{valid_count}/{len(repos)} working"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "repos": [], "value": "Error"}

def find_specialization(pdf_path):
    specializations = {
        "Software Technology": "software technology",
        "Network Technology": "network technology",
        "Multimedia Technology": "multimedia technology"
    }

    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(min(3, len(doc))):
            text = doc[page_num].get_text().lower()
            for spec_name, keyword in specializations.items():
                if keyword in text:
                    doc.close()
                    return {
                        "status": "success",
                        "message": f"Specialization: {spec_name}",
                        "value": spec_name
                    }
        doc.close()
        return {
            "status": "warning",
            "message": "Specialization area එක clearly mention කරලා නෑ.",
            "value": "Not specified"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}", "value": "Error"}

def validate_with_llm(pdf_path):
    try:
        markdown_text = pymupdf4llm.to_markdown(pdf_path)

        query = f'''
{markdown_text}

Just say yes or no to these questions:
(1) Does the O/L A/L Results have mention here?
(2) Our degree name is "Bachelor of Information and Communication Technology (Hons)" Does it mentioned clearly?
(3) Does certificate mentioned here?
(4) Are technical & soft skills separated?
(5) Do projects include technologies used?
(6) Is grammar & spelling correct?
(7) Are proper section titles used?
(8) Is there a valid references section?
'''

        groq_api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")

        if not groq_api_key:
            return {
                "status": "error",
                "message": "API key නැහැ. config.py file එකේ GROQ_API_KEY එක set කරන්න.",
                "results": [],
                "passed": 0,
                "total": 8
            }

        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        start_time = time.time()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="openai/gpt-oss-120b",
            temperature=0.1
        )
        execution_time = time.time() - start_time

        answers = response.choices[0].message.content.strip().splitlines()

        criteria = [
            ("O/L & A/L Results", "Academic qualifications mention කරලා තියෙනවද"),
            ("Degree Name", "Degree එකේ නම හරියට Mention කරලා තියෙනවාද?"),
            ("Certificates", "Certificates section එකක් තියෙනවද"),
            ("Skills Separation", "Technical/Soft skills වෙන් කරලා තියෙනවද"),
            ("Project Technologies", "Projects වල technologies mention කරලා තියෙනවද"),
            ("Grammar & Spelling", "Grammar සහ spelling හරිද"),
            ("Section Titles", "Proper headings use කරලා තියෙනවද"),
            ("References", "References section එකක් තියෙනවද")
        ]

        llm_results = []
        passed = 0

        for i, ans in enumerate(answers[:len(criteria)]):
            answer_clean = ans.strip().lower()
            is_yes = 'yes' in answer_clean or answer_clean == 'y'

            if is_yes:
                passed += 1

            llm_results.append({
                "name": criteria[i][0],
                "description": criteria[i][1],
                "passed": is_yes
            })

        return {
            "status": "success",
            "message": f"AI Analysis done ({execution_time:.1f}s)",
            "results": llm_results,
            "passed": passed,
            "total": len(criteria)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"AI Analysis error: {str(e)}",
            "results": [],
            "passed": 0,
            "total": 8
        }

# ============= HTML Template =============

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="si">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CV Validator</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Sinhala:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Noto Sans Sinhala', 'Segoe UI', sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 40px 20px;
            background: white;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .header h1 {
            font-size: 28px;
            color: #2c3e50;
            margin-bottom: 8px;
        }

        .header p {
            color: #7f8c8d;
            font-size: 15px;
        }

        /* Upload Box */
        .upload-box {
            background: white;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px 20px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .upload-area:hover {
            border-color: #3498db;
            background: #f8fbff;
        }

        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 16px;
            color: #555;
            margin-bottom: 5px;
        }

        .upload-hint {
            font-size: 13px;
            color: #999;
        }

        .file-input {
            display: none;
        }

        .selected-file {
            display: none;
            margin-top: 20px;
            padding: 12px 20px;
            background: #e8f4fd;
            border-radius: 8px;
            color: #2980b9;
            font-size: 14px;
        }

        .selected-file.show {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .clear-btn {
            background: none;
            border: none;
            color: #e74c3c;
            cursor: pointer;
            font-size: 18px;
            padding: 0 5px;
        }

        .submit-btn {
            margin-top: 25px;
            padding: 14px 40px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            cursor: pointer;
            transition: background 0.2s;
        }

        .submit-btn:hover {
            background: #2980b9;
        }

        .submit-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }

        /* Results */
        .results-header {
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .results-title {
            font-size: 22px;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .results-file {
            color: #7f8c8d;
            font-size: 14px;
        }

        /* Summary Stats */
        .summary-stats {
            display: flex;
            gap: 15px;
            margin-top: 15px;
        }

        .stat-box {
            flex: 1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-box.issues {
            background: #fff3cd;
            border: 1px solid #ffc107;
        }

        .stat-box.passed {
            background: #d4edda;
            border: 1px solid #28a745;
        }

        .stat-number {
            font-size: 28px;
            font-weight: 600;
        }

        .stat-box.issues .stat-number { color: #856404; }
        .stat-box.passed .stat-number { color: #28a745; }

        .stat-label {
            font-size: 13px;
            color: #666;
            margin-top: 3px;
        }

        /* Check Items */
        .check-section {
            background: white;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }

        .check-header {
            padding: 18px 20px;
            font-weight: 600;
            font-size: 15px;
            background: #fafbfc;
            border-bottom: 1px solid #eee;
            color: #2c3e50;
        }

        .check-header.issues-header {
            background: #fff8e6;
            color: #856404;
        }

        .check-item {
            display: flex;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid #f0f0f0;
        }

        .check-item:last-child {
            border-bottom: none;
        }

        .check-status {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 14px;
            flex-shrink: 0;
        }

        .check-status.pass {
            background: #d4edda;
            color: #28a745;
        }

        .check-status.warn {
            background: #fff3cd;
            color: #856404;
        }

        .check-status.fail {
            background: #f8d7da;
            color: #dc3545;
        }

        .check-content {
            flex: 1;
        }

        .check-name {
            font-weight: 500;
            color: #2c3e50;
            margin-bottom: 3px;
        }

        .check-message {
            font-size: 13px;
            color: #7f8c8d;
        }

        .check-value {
            font-size: 13px;
            color: #555;
            background: #f5f5f5;
            padding: 4px 10px;
            border-radius: 4px;
            margin-left: 10px;
        }

        /* GitHub Links */
        .repo-list {
            padding: 0 20px 15px;
        }

        .repo-item {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 13px;
        }

        .repo-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
        }

        .repo-dot.valid { background: #28a745; }
        .repo-dot.invalid { background: #dc3545; }

        .repo-url {
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }

        .repo-url:hover {
            text-decoration: underline;
        }

        /* Dropdown */
        .dropdown {
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .dropdown-header {
            background: #f0fff4;
            padding: 18px 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border: 1px solid #c3e6cb;
            border-radius: 12px;
            transition: all 0.2s;
        }

        .dropdown-header:hover {
            background: #e6f7eb;
        }

        .dropdown-header.open {
            border-radius: 12px 12px 0 0;
            border-bottom: none;
        }

        .dropdown-title {
            font-weight: 600;
            color: #28a745;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .dropdown-count {
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 13px;
        }

        .dropdown-arrow {
            font-size: 14px;
            color: #28a745;
            transition: transform 0.2s;
        }

        .dropdown-header.open .dropdown-arrow {
            transform: rotate(180deg);
        }

        .dropdown-content {
            display: none;
            background: white;
            border: 1px solid #c3e6cb;
            border-top: none;
            border-radius: 0 0 12px 12px;
        }

        .dropdown-content.show {
            display: block;
        }

        /* No issues message */
        .no-issues {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            margin-bottom: 15px;
        }

        .no-issues-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }

        .no-issues-text {
            color: #28a745;
            font-weight: 500;
            font-size: 16px;
        }

        /* Back Button */
        .back-section {
            text-align: center;
            padding: 30px;
        }

        .back-btn {
            display: inline-block;
            padding: 12px 30px;
            background: white;
            color: #3498db;
            text-decoration: none;
            border-radius: 8px;
            font-size: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.2s;
        }

        .back-btn:hover {
            background: #3498db;
            color: white;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 13px;
        }

        /* Loading */
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.95);
            z-index: 100;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .loading.show {
            display: flex;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #eee;
            border-top-color: #3498db;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-text {
            margin-top: 15px;
            color: #666;
        }

        /* Mobile */
        @media (max-width: 600px) {
            .container { padding: 15px; }
            .header { padding: 30px 15px; }
            .header h1 { font-size: 24px; }
            .upload-box { padding: 25px; }
            .upload-area { padding: 30px 15px; }
            .check-item { padding: 14px 15px; }
            .summary-stats { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 CV Validator</h1>
            <p>ඔයාගේ CV එක check කරලා feedback ගන්න</p>
        </div>

        {% if not results %}
        <div class="upload-box">
            <form method="POST" enctype="multipart/form-data" id="uploadForm">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">PDF file එක මෙතන click කරලා select කරන්න</div>
                    <div class="upload-hint">හෝ drag & drop කරන්න</div>
                    <input type="file" name="cv_file" accept=".pdf" class="file-input" id="fileInput" required>
                </div>

                <div class="selected-file" id="selectedFile">
                    <span>📎</span>
                    <span id="fileName"></span>
                    <button type="button" class="clear-btn" id="clearBtn">✕</button>
                </div>

                <button type="submit" class="submit-btn" id="submitBtn" disabled>
                    🔍 CV එක Check කරන්න
                </button>
            </form>
        </div>
        {% endif %}

        {% if results %}
        {% set all_checks = [] %}
        {% set issues = [] %}
        {% set passed = [] %}

        {# Basic checks #}
        {% set basic_checks = [
            ('Page Count', results.page_count),
            ('GPA / CGPA', results.gpa),
            ('Specialization', results.specialization),
            ('GitHub Links', results.github)
        ] %}

        {% for name, check in basic_checks %}
            {% if check.status == 'success' %}
                {% set _ = passed.append({'name': name, 'message': check.message, 'value': check.value, 'type': 'basic', 'github_repos': check.get('repos', [])}) %}
            {% else %}
                {% set _ = issues.append({'name': name, 'message': check.message, 'value': check.value, 'status': check.status, 'type': 'basic', 'github_repos': check.get('repos', [])}) %}
            {% endif %}
        {% endfor %}

        {# AI checks #}
        {% if results.llm.status == 'success' %}
            {% for item in results.llm.results %}
                {% if item.passed %}
                    {% set _ = passed.append({'name': item.name, 'message': item.description, 'type': 'ai'}) %}
                {% else %}
                    {% set _ = issues.append({'name': item.name, 'message': item.description, 'status': 'warning', 'type': 'ai'}) %}
                {% endif %}
            {% endfor %}
        {% endif %}

        <div class="results-header">
            <h2 class="results-title">📊 Validation Results</h2>
            <p class="results-file">File: {{ filename }}</p>

            <div class="summary-stats">
                <div class="stat-box issues">
                    <div class="stat-number">{{ issues|length }}</div>
                    <div class="stat-label"> ඔයාගේ CV එකේ, ඉක්මනින් අවධානය යොමු කල යුතු ගැටළු ගණන </div>
                </div>

            </div>
        </div>

        <!-- Issues Section (Shown Prominently) -->
        {% if issues|length > 0 %}
        <div class="check-section">
            <div class="check-header issues-header"> හදන්න ඕන දේවල් ({{ issues|length }})</div>

            {% for item in issues %}
            <div class="check-item">
                <div class="check-status {% if item.status == 'warning' %}warn{% else %}fail{% endif %}">
                    {% if item.status == 'warning' %}!{% else %}✕{% endif %}
                </div>
                <div class="check-content">
                    <div class="check-name">{{ item.name }}</div>
                    <div class="check-message">{{ item.message }}</div>
                </div>
                {% if item.value %}
                <div class="check-value">{{ item.value }}</div>
                {% endif %}
            </div>

            {% if item.name == 'GitHub Links' and item.github_repos %}
            <div class="repo-list">
                {% for repo in item.github_repos %}
                <div class="repo-item">
                    <span class="repo-dot {% if repo.valid %}valid{% else %}invalid{% endif %}"></span>
                    <a href="{{ repo.url }}" target="_blank" class="repo-url">{{ repo.url }}</a>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endfor %}
        </div>
        {% else %}
        <div class="no-issues">
            <div class="no-issues-icon">🎉</div>
            <div class="no-issues-text">සුපිරි! ඔයාගේ CV එකේ issues නෑ!</div>
        </div>
        {% endif %}

        <!-- Passed Items (Dropdown) -->
        {% if passed|length > 0 %}
        <div class="dropdown">
            <div class="dropdown-header" id="passedDropdown">
                <div class="dropdown-title">
                    ✅ හරියට තියෙන දේවල්
                    <span class="dropdown-count">{{ passed|length }}</span>
                </div>
                <span class="dropdown-arrow">▼</span>
            </div>
            <div class="dropdown-content" id="passedContent">
                {% for item in passed %}
                <div class="check-item">
                    <div class="check-status pass">✓</div>
                    <div class="check-content">
                        <div class="check-name">{{ item.name }}</div>
                        <div class="check-message">{{ item.message }}</div>
                    </div>
                    {% if item.value %}
                    <div class="check-value">{{ item.value }}</div>
                    {% endif %}
                </div>

                {% if item.name == 'GitHub Links' and item.github_repos %}
                <div class="repo-list">
                    {% for repo in item.github_repos %}
                    <div class="repo-item">
                        <span class="repo-dot {% if repo.valid %}valid{% else %}invalid{% endif %}"></span>
                        <a href="{{ repo.url }}" target="_blank" class="repo-url">{{ repo.url }}</a>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if results.llm.status == 'error' %}
        <div class="check-section">
            <div class="check-header">🤖 AI Analysis</div>
            <div class="check-item">
                <div class="check-status fail">!</div>
                <div class="check-content">
                    <div class="check-name">Error</div>
                    <div class="check-message">{{ results.llm.message }}</div>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="back-section">
            <a href="/" class="back-btn">← තවත් CV එකක් Check කරන්න</a>
        </div>
        {% endif %}

        <div class="footer">
            Made by Janith | CV Validator v2.0
        </div>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div class="loading-text">CV එක analyze කරමින්...</div>
    </div>

    <script>
        // Upload functionality
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const selectedFile = document.getElementById('selectedFile');
        const fileName = document.getElementById('fileName');
        const clearBtn = document.getElementById('clearBtn');
        const submitBtn = document.getElementById('submitBtn');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');

        if (uploadArea) {
            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#3498db';
                uploadArea.style.background = '#f8fbff';
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#ddd';
                uploadArea.style.background = '';
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ddd';
                uploadArea.style.background = '';

                const files = e.dataTransfer.files;
                if (files.length > 0 && files[0].type === 'application/pdf') {
                    fileInput.files = files;
                    showFile(files[0].name);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    showFile(e.target.files[0].name);
                }
            });

            clearBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.value = '';
                selectedFile.classList.remove('show');
                submitBtn.disabled = true;
            });

            uploadForm.addEventListener('submit', () => {
                loading.classList.add('show');
            });
        }

        function showFile(name) {
            fileName.textContent = name;
            selectedFile.classList.add('show');
            submitBtn.disabled = false;
        }

        // Dropdown functionality
        const passedDropdown = document.getElementById('passedDropdown');
        const passedContent = document.getElementById('passedContent');

        if (passedDropdown) {
            passedDropdown.addEventListener('click', () => {
                passedDropdown.classList.toggle('open');
                passedContent.classList.toggle('show');
            });
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'cv_file' not in request.files:
            return 'කරුණාකර CV file එකක් select කරන්න'

        file = request.files['cv_file']

        if file.filename == '':
            return 'කරුණාකර CV file එකක් select කරන්න'

        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = {
                'page_count': check_cv_page_count(filepath),
                'gpa': check_gpa_in_cv(filepath),
                'specialization': find_specialization(filepath),
                'github': validate_github_links(filepath),
                'llm': validate_with_llm(filepath)
            }

            os.remove(filepath)

            return render_template_string(HTML_TEMPLATE, results=results, filename=filename)
        else:
            return 'කරුණාකර PDF file එකක් විතරක් upload කරන්න'

    return render_template_string(HTML_TEMPLATE, results=None)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CV Validator")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
