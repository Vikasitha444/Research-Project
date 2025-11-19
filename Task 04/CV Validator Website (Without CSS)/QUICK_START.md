# CV Validator - Quick Start Guide
## ජනිත් විසින් හදන ලද

---

## 🚀 ලේසිම විදිහට Run කරන්න:

### Windows Users:
```
run_app.bat file එක double-click කරන්න
```

### Linux/Mac Users:
```bash
./run_app.sh
```

---

## 📋 Manual Setup (First Time Only):

### Step 1: Dependencies Install කරන්න
```bash
pip install -r requirements.txt
```

### Step 2: Application එක Run කරන්න
```bash
python cv_validator_app.py
```

### Step 3: Browser Open කරන්න
```
http://127.0.0.1:5000
```

---

## ✅ මේකෙන් Check වෙන දේවල්:

### Basic Python Validations:
1. ✓ **Page Count** - CV එකේ pages 1ක් විතරක් තියෙනවද?
2. ✓ **GPA Check** - GPA/CGPA mention කරලා තියෙනවද?
3. ✓ **Specialization** - Software/Network/Multimedia Technology එක mention කරලා තියෙනවද?
4. ✓ **GitHub Links** - GitHub repository links තියෙනවද? වැඩ කරනවද?

### Advanced AI-Powered Validations (LLM):
5. ✓ **O/L & A/L Results** - Educational qualifications properly mentioned ද?
6. ✓ **Correct Degree Name** - "Bachelor of ICT (Hons)" හරියට තියෙනවද?
7. ✓ **Certificates Section** - Certificates mention කරලා තියෙනවද?
8. ✓ **Skills Separation** - Technical & soft skills වෙන වෙනමද?
9. ✓ **Project Technologies** - Projects වල tech stack mention කරලා තියෙනවද?
10. ✓ **Grammar & Spelling** - Language quality හොඳද?
11. ✓ **Section Titles** - Proper headings use කරලා තියෙනවද?
12. ✓ **References Section** - Valid references තියෙනවද?

---

## 📁 Project Files:

- **cv_validator_app.py** - Main application file
- **requirements.txt** - Python packages list
- **run_app.bat** - Windows run script
- **run_app.sh** - Linux/Mac run script
- **INSTRUCTIONS_SINHALA.md** - විස්තරාත්මක instructions

---

## 🎯 භාවිතය:

1. Application එක run කරන්න
2. Browser එකෙන් open කරන්න (http://127.0.0.1:5000)
3. CV file එක (PDF) upload කරන්න
4. "CV එක Validate කරන්න" button එක click කරන්න
5. Results එක බලන්න!

---

## 🔧 Common Issues:

**"Module not found" error:**
```bash
pip install Flask pymupdf pymupdf4llm requests
```

**"Port 5000 already in use":**
- cv_validator_app.py file එකේ අන්තිම line එකේ port number එක වෙනස් කරන්න

---

## 🎨 Features:

- ✨ Simple HTML interface
- 🔒 Secure file handling
- 🗑️ Auto cleanup (uploaded files delete වෙනවා)
- ⚡ Fast validation
- 🤖 AI-powered advanced checks (Groq LLM)
- 📊 Detailed results with color-coded status
- 🌐 Web-based (any device එකෙන් use කරන්න පුළුවන්)
- 🎯 12 different validation checks (4 Python + 8 AI)

---

## 🔮 Future Improvements:

- CSS එක add කරලා modern UI එකක් හදන්න
- Email validation එකතු කරන්න
- LinkedIn profile check කරන්න
- PDF report generate කරන්න
- Multiple CVs batch processing

---

## 📞 Support:

Issues හම්බුණොත් හෝ suggestions තියෙනවා නම්:
- GitHub repository එකට issue එකක් දාන්න
- Team එකට message එකක් යවන්න

---

**විශේෂ සටහන:** මේ application එක තමයි ඔයාගේ සියලු CV validation tasks එකතු කරලා web interface එකක් හරහා run කරන්න හදපු එක. Flask භාවිතා කරන නිසා simple සහ lightweight යි!

**සාර්ථක CV Validation එකක් වේවා! 🎉**
