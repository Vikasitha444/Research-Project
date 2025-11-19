from flask import Flask, request, render_template_string
import os
import pymupdf
import pymupdf4llm
import re
import requests
import time
from werkzeug.utils import secure_filename
from openai import OpenAI

# Config file එකෙන් API key ගන්න try කරනවා
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Upload folder එක හදාගන්නවා
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============= Validation Functions =============

def check_cv_page_count(pdf_path):
    """CV එකේ pages ගණන check කරනවා"""
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        if page_count > 1:
            return {
                "status": "warning",
                "message": f"CV එකේ pages {page_count}ක් තියෙනවා! උපරිම තිබිය හැක්කේ එක page එකක් පමණයි."
            }
        else:
            return {
                "status": "success",
                "message": f"CV එක page {page_count}කින් තියෙනවා - Perfect!"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {e}"
        }

def check_gpa_in_cv(pdf_path):
    """GPA එක mention කරලා තියෙනවද check කරනවා"""
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
                "message": "GPA එක mention කරලා තියෙනවා!"
            }
        else:
            return {
                "status": "warning",
                "message": "GPA එක mention කරලා නෑ! Suggestion: Add current GPA, if you have good results"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {e}"
        }

def extract_github_links(text):
    """GitHub repository links විතරක් Extract කරනවා"""
    repo_pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
    repos = re.findall(repo_pattern, text)
    return list(set(repos))

def check_repository_exists(repo_url):
    """GitHub repository එක තියෙනවද check කරනවා"""
    try:
        response = requests.head(repo_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return True, "Repository තියෙනවා"
        elif response.status_code == 404:
            return False, "Repository එක හම්බෙන්නේ නැහැ (404)"
        else:
            return False, f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout - Server respond කරේ නැහැ"
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {str(e)}"

def validate_github_links(pdf_path):
    """GitHub links validate කරනවා"""
    try:
        markdown = pymupdf4llm.to_markdown(pdf_path)
        repos = extract_github_links(markdown)
        
        if not repos:
            return {
                "status": "warning",
                "message": "GitHub Repository links හම්බෙන්නේ නැහැ!",
                "repos": []
            }
        
        valid_count = 0
        invalid_count = 0
        repo_details = []
        
        for repo in repos:
            exists, message = check_repository_exists(repo)
            repo_details.append({
                "url": repo,
                "valid": exists,
                "message": message
            })
            
            if exists:
                valid_count += 1
            else:
                invalid_count += 1
            
            time.sleep(0.5)
        
        if valid_count == len(repos):
            status = "success"
        elif valid_count > 0:
            status = "warning"
        else:
            status = "error"
        
        return {
            "status": status,
            "message": f"හම්බුණු Repositories: {len(repos)} | Valid: {valid_count} | Invalid: {invalid_count}",
            "repos": repo_details
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {e}",
            "repos": []
        }

def find_specialization(pdf_path):
    """Specialization area එක හොයනවා"""
    specializations = {
        "Software Technology": "software technology",
        "Network Technology": "network technology", 
        "Multimedia Technology": "multimedia technology"
    }
    
    try:
        doc = pymupdf.open(pdf_path)
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text().lower()
            
            for spec_name, keyword in specializations.items():
                if keyword in text:
                    doc.close()
                    return {
                        "status": "success",
                        "message": f"Specialization: {spec_name} (Page {page_num + 1} එකේ හම්බුණා)"
                    }
        
        doc.close()
        return {
            "status": "warning",
            "message": "Specialization area එකක් හම්බුණේ නැහැ"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {e}"
        }

def validate_with_llm(pdf_path):
    """LLM Model එක use කරලා additional criteria check කරනවා"""
    try:
        # PDF එක markdown format එකට convert කරනවා
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Query එක prepare කරනවා
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

        # Groq API key set කරනවා (priority: config.py > environment variable > hardcoded)
        groq_api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "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")
        
        # OpenAI client එක Groq API එක සමග setup කරනවා
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        
        # API call එක කරනවා
        start_time = time.time()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="openai/gpt-oss-120b",  # Groq model එකක් use කරනවා
            temperature=0.1  # Consistent answers වලට
        )
        end_time = time.time()
        
        # Response parse කරනවා
        answers = response.choices[0].message.content.strip().splitlines()
        
        criteria = [
            "Does the O/L A/L Results have mention here?",
            "Is the degree name 'Bachelor of Information and Communication Technology (Hons)' mentioned clearly?",
            "Does certificate mentioned here?",
            "Are technical & soft skills separated?",
            "Do projects include technologies used?",
            "Is grammar & spelling correct?",
            "Are proper section titles used?",
            "Is there a valid references section?"
        ]
        
        # Results format කරනවා
        llm_results = []
        for i, ans in enumerate(answers[:len(criteria)]):
            answer_clean = ans.strip().lower()
            
            # Yes/No detect කරනවා
            if 'yes' in answer_clean or 'y' == answer_clean:
                status = "success"
                result = "Yes ✓"
            elif 'no' in answer_clean or 'n' == answer_clean:
                status = "warning"
                result = "No ✗"
            else:
                status = "info"
                result = ans.strip()
            
            llm_results.append({
                "criteria": criteria[i],
                "result": result,
                "status": status
            })
        
        execution_time = end_time - start_time
        
        return {
            "status": "success",
            "message": f"LLM Validation completed in {execution_time:.2f} seconds",
            "results": llm_results,
            "execution_time": execution_time
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"LLM Validation Error: {str(e)}",
            "results": [],
            "execution_time": 0
        }

# ============= HTML Template =============

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CV Validator - හදන්න ලද ජනිත්</title>
</head>
<body>
    <h1>CV Validator System</h1>
    <h2>CV එකක් Upload කරන්න</h2>
    
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="cv_file" accept=".pdf" required>
        <br><br>
        <button type="submit">CV එක Validate කරන්න</button>
    </form>
    
    {% if results %}
    <hr>
    <h2>Validation Results</h2>
    
    <h3>1. Page Count Check</h3>
    <p><strong>Status:</strong> {{ results.page_count.status }}</p>
    <p>{{ results.page_count.message }}</p>
    
    <hr>
    
    <h3>2. GPA Check</h3>
    <p><strong>Status:</strong> {{ results.gpa.status }}</p>
    <p>{{ results.gpa.message }}</p>
    
    <hr>
    
    <h3>3. Specialization Check</h3>
    <p><strong>Status:</strong> {{ results.specialization.status }}</p>
    <p>{{ results.specialization.message }}</p>
    
    <hr>
    
    <h3>4. GitHub Links Validation</h3>
    <p><strong>Status:</strong> {{ results.github.status }}</p>
    <p>{{ results.github.message }}</p>
    
    {% if results.github.repos %}
    <h4>Repository Details:</h4>
    <ul>
        {% for repo in results.github.repos %}
        <li>
            <strong>{{ repo.url }}</strong><br>
            {% if repo.valid %}
            ✓ {{ repo.message }}
            {% else %}
            ✗ {{ repo.message }}
            {% endif %}
        </li>
        {% endfor %}
    </ul>
    {% endif %}
    
    <hr>
    
    <h3>5. LLM-Based Advanced Validation</h3>
    <p><strong>Status:</strong> {{ results.llm.status }}</p>
    <p>{{ results.llm.message }}</p>
    
    {% if results.llm.results %}
    <table border="1" cellpadding="10" cellspacing="0">
        <thead>
            <tr>
                <th>No.</th>
                <th>Criteria</th>
                <th>Result</th>
            </tr>
        </thead>
        <tbody>
            {% for item in results.llm.results %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ item.criteria }}</td>
                <td>
                    {% if item.status == 'success' %}
                    <strong style="color: green;">{{ item.result }}</strong>
                    {% elif item.status == 'warning' %}
                    <strong style="color: orange;">{{ item.result }}</strong>
                    {% else %}
                    {{ item.result }}
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <p><em>Execution Time: {{ "%.2f"|format(results.llm.execution_time) }} seconds</em></p>
    {% endif %}
    
    <hr>
    
    <h3>Summary</h3>
    <p>CV File: {{ filename }}</p>
    <p>Validation Complete!</p>
    
    <br>
    <a href="/">තවත් CV එකක් Check කරන්න</a>
    {% endif %}
</body>
</html>
'''

# ============= Routes =============

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # File upload එක handle කරනවා
        if 'cv_file' not in request.files:
            return 'කරුණාකර CV file එකක් select කරන්න'
        
        file = request.files['cv_file']
        
        if file.filename == '':
            return 'කරුණාකර CV file එකක් select කරන්න'
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # සියලු validations run කරනවා
            results = {
                'page_count': check_cv_page_count(filepath),
                'gpa': check_gpa_in_cv(filepath),
                'specialization': find_specialization(filepath),
                'github': validate_github_links(filepath),
                'llm': validate_with_llm(filepath)  # LLM validation එකත් add කරනවා
            }
            
            # File එක delete කරනවා (cleanup)
            os.remove(filepath)
            
            return render_template_string(HTML_TEMPLATE, results=results, filename=filename)
        else:
            return 'කරුණාකර PDF file එකක් විතරක් upload කරන්න'
    
    return render_template_string(HTML_TEMPLATE, results=None)

if __name__ == '__main__':
    print("=" * 60)
    print("CV Validator Web Application")
    print("Server running on: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
