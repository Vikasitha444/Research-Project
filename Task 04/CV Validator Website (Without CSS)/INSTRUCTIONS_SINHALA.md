# CV Validator Web Application
## හදන්න ලද - ජනිත්

### මෙය භාවිතා කරන්නේ කොහොමද?

#### 1. Requirements Install කරන්න:
```bash
pip install Flask pymupdf pymupdf4llm requests openai
```

හෝ requirements.txt එක use කරලා:
```bash
pip install -r requirements.txt
```

#### 2. (Optional) Groq API Key Setup:
LLM-based validation feature එක use කරන්න නම් Groq API key එකක් ඕන.

**Option 1: config.py file එක edit කරන්න**
```python
GROQ_API_KEY = "your_api_key_here"
```

**Option 2: Environment variable set කරන්න**
```bash
# Windows
set GROQ_API_KEY=your_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_api_key_here
```

Free API key එකක් ලබාගන්න: https://console.groq.com/keys

#### 3. Application එක Run කරන්න:
```bash
python cv_validator_app.py
```

#### 4. Browser එකෙන් Open කරන්න:
```
http://127.0.0.1:5000
```

### මේකෙන් Check වෙන දේවල්:

#### Basic Validations (Python-based):
1. **Page Count** - CV එකේ pages 1ක් විතරක් තියෙනවද?
2. **GPA Mentioned** - GPA එක mention කරලා තියෙනවද?
3. **Specialization Area** - Software/Network/Multimedia Technology mention කරලා තියෙනවද?
4. **GitHub Links** - GitHub links තියෙනවද? ඒවා valid links ද?

#### Advanced LLM Validation (AI-powered):
5. **O/L & A/L Results** - Educational qualifications mention කරලා තියෙනවද?
6. **Correct Degree Name** - "Bachelor of Information and Communication Technology (Hons)" properly mention කරලා තියෙනවද?
7. **Certificates** - Certificates section එක තියෙනවද?
8. **Skills Separation** - Technical & soft skills වෙන වෙනම organize කරලා තියෙනවද?
9. **Project Technologies** - Projects වල use කරපු technologies mention කරලා තියෙනවද?
10. **Grammar & Spelling** - Grammar සහ spelling හරිද?
11. **Section Titles** - Proper section headings use කරලා තියෙනවද?
12. **References** - Valid references section එකක් තියෙනවද?

### Features:

- Simple HTML interface (No CSS)
- PDF files විතරක් upload කරන්න පුළුවන්
- Maximum file size: 16MB
- Python-based + AI-powered validations
- Results එක details සමග පෙන්වනවා
- Upload කරපු file automatically delete වෙනවා (security)
- Groq AI model භාවිතා කරලා advanced checks

### Project Structure:

```
cv_validator_app.py     - Main Flask application
config.py               - API key configuration
requirements.txt        - Python dependencies
uploads/               - Temporary upload folder (auto-created)
```

### Troubleshooting:

1. **Port 5000 already in use** error එනවා නම්:
   - Code එකේ අන්තිම line එකේ `port=5000` එක වෙනස් කරන්න (උදා: `port=5001`)

2. **Module not found** errors:
   - කරුණාකර requirements install කරන්න: `pip install -r requirements.txt`

3. **Upload folder issues**:
   - Application එක automatically "uploads" folder එක හදයි
   - Manual ගාලා හදන්න ඕන නෑ

4. **LLM validation errors**:
   - Groq API key එක properly set කරලා තියෙනවද check කරන්න
   - Internet connection එක තියෙනවද verify කරන්න
   - Free tier limits exceed වෙලා නැද්ද බලන්න

### API Rate Limits:

Groq free tier එකේ:
- Daily request limit තියෙනවා
- Limit එක exceed වුනොත්, basic validations විතරක් run වෙනවා
- LLM validation එක fail වුනත්, අනිත් validations වැඩ කරනවා

### Next Steps (Future Improvements):

- CSS එක add කරලා UI එක enhance කරන්න පුළුවන්
- More validation checks එකතු කරන්න පුළුවන්
- Results එක PDF report එකක් විදිහට download කරන්න පුළුවන්
- Multiple CVs එකවර upload කරන්න පුළුවන්
- Different AI models test කරන්න පුළුවන්

### Contact:

Any issues or suggestions - අපේ project එකට contribute කරන්න!
