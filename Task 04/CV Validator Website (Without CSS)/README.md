# CV Validator with AI-Powered Analysis
## 🎓 University Research Project by Janith

---

## 🌟 Overview

මේ CV Validator System එක හදන ලද්දේ, University students ගේ CV වල තියෙන common mistakes හොයාගෙන, suggestions දෙන්න. මේකේ unique වෙන ඒ හැම මොහොතම Python validations වගේම **AI-powered LLM validation** එකත් තියෙන එක!

---

## 🚀 Key Features

### 1. **Python-Based Validations** (Fast & Offline)
- Page count verification
- GPA/CGPA detection
- Specialization area identification
- GitHub repository link validation with live status checking

### 2. **AI-Powered LLM Validations** (Groq API)
- Advanced content analysis using Large Language Models
- 8 comprehensive criteria checks
- Natural language understanding
- Context-aware feedback

---

## 📋 Complete Validation List

| No. | Validation Type | Powered By | Description |
|-----|----------------|------------|-------------|
| 1 | Page Count | Python | CV එකේ pages 1ක් විතරද? |
| 2 | GPA Mentioned | Python | GPA/CGPA mention කරලා තියෙනවද? |
| 3 | Specialization | Python | Software/Network/Multimedia Technology එක තියෙනවද? |
| 4 | GitHub Links | Python | GitHub links valid ද? Live status checking |
| 5 | O/L A/L Results | AI (LLM) | Educational qualifications properly mentioned ද? |
| 6 | Degree Name | AI (LLM) | "Bachelor of ICT (Hons)" හරියට තියෙනවද? |
| 7 | Certificates | AI (LLM) | Certificates section එක තියෙනවද? |
| 8 | Skills Separation | AI (LLM) | Technical & soft skills වෙන වෙනමද? |
| 9 | Project Tech | AI (LLM) | Projects වල technologies mention කරලා තියෙනවද? |
| 10 | Grammar | AI (LLM) | Grammar සහ spelling හොඳද? |
| 11 | Section Titles | AI (LLM) | Proper headings use කරලා තියෙනවද? |
| 12 | References | AI (LLM) | Valid references section තියෙනවද? |

---

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **PDF Processing**: PyMuPDF, pymupdf4llm
- **AI/LLM**: Groq API (llama3-8b-8192 model)
- **HTTP Requests**: requests library
- **Frontend**: Simple HTML (No CSS - intentional)

---

## 📦 Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Groq API Key

**Option A: Edit config.py file**
```python
GROQ_API_KEY = "your_groq_api_key_here"
```

**Option B: Use Environment Variable**
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_groq_api_key_here
```

**Get Free API Key:** https://console.groq.com/keys

### Step 3: Run the Application
```bash
python cv_validator_app.py
```

### Step 4: Open in Browser
```
http://127.0.0.1:5000
```

---

## 📱 Usage

1. **Open the web interface**
2. **Upload your CV** (PDF format only)
3. **Click "CV එක Validate කරන්න"**
4. **Review results** with detailed feedback

---

## 🎯 Project Structure

```
cv_validator/
│
├── cv_validator_app.py    # Main Flask application
├── config.py              # API key configuration
├── requirements.txt       # Python dependencies
├── uploads/               # Temporary upload folder (auto-created)
├── QUICK_START.md        # Quick reference guide
├── INSTRUCTIONS_SINHALA.md # Detailed Sinhala instructions
└── README.md             # This file
```

---

## 🔍 How It Works

### Python Validations:
1. Upload PDF → Save temporarily
2. Extract text using PyMuPDF
3. Run regex patterns & keyword searches
4. Check GitHub links with live HTTP requests
5. Return structured results

### LLM Validation:
1. Convert PDF to markdown (pymupdf4llm)
2. Create structured prompt with CV content
3. Send to Groq API (llama3-8b model)
4. Parse AI response (Yes/No answers)
5. Display with color-coded status

---

## ⚠️ Important Notes

### API Rate Limits:
- Groq free tier එකේ daily limits තියෙනවා
- LLM validation fail වුනත්, Python validations වැඩ කරනවා
- Internet connection එක ඕන LLM validation වලට

### File Security:
- Upload කරපු files automatically delete වෙනවා
- No data retention

---

## 🔮 Future Enhancements

- Modern UI with CSS/Tailwind
- Multiple CV batch processing
- PDF report generation
- Email validation
- LinkedIn profile checking
- ATS optimization

---

## 🐛 Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"Port 5000 already in use":**
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

**LLM validation errors:**
- Check API key configuration
- Verify internet connection
- Check Groq API limits

---

## 👥 Contributors

**Project Lead:** Janith
- IT Manager @ Institute of English Kolonnawa
- HND in Cyber Security student

**University:** University of Sri Jayewardenepura

---

**සාර්ථක CV Validation එකක් වේවා! 🎉**
