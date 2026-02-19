# 🎯 CV Validator & Job Recommendation System

CV Validator and Job Recommendation System tailored for the Sri Lankan IT Market 🇱🇰

## ✨ Features

### CV Validation
- ✅ Page count check (1-2 pages recommended)
- ✅ GPA detection
- ✅ Professional email validation
- ✅ Photo presence (recommended for Sri Lankan market)
- ✅ O/L & A/L results check
- ✅ Formatting quality analysis
- ✅ Technical keywords extraction (Python, Java, React, etc.)
- ✅ GitHub links validation
- ✅ Skills separation (Technical vs Soft skills)
- ✅ Contact information completeness
- ✅ Action verbs usage
- ✅ Quantifiable achievements
- ✅ Professional summary/objective
- ✅ ATS compatibility check
- ✅ Consistency analysis
- ✅ AI-powered HR Manager analysis

### Job Recommendation
- 💼 Automatic job matching based on CV keywords
- 📊 Match percentage calculation
- 🎯 Jobs ranked by relevance
- 📍 Location-based filtering
- 🔗 Direct application links

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Key

Edit `config.py` and add your Groq API key:

```python
GROQ_API_KEY = "your-actual-groq-api-key"
```

To get a free Groq API key:
1. Visit https://console.groq.com
2. Sign up for a free account
3. Generate an API key

### 3. Add Job Data

Make sure you have `topjobs_it_jobs.csv` in the same directory with the following columns:
- title
- company
- location
- description
- url (optional)
- closing_date (optional)

## 📖 Usage

### Run the Application

```bash
python cv_validator_with_jobs.py
```

The application will start at `http://127.0.0.1:5000`

### Using the Interface

1. **Upload CV**: Drag and drop or click to browse your PDF CV
2. **Validate**: Click "Validate CV & Find Jobs" button
3. **View Results**:
   - **CV Validation Tab**: See detailed validation results
   - **Job Recommendations Tab**: Browse matching jobs
   - **Your Skills Tab**: View detected technical skills

### API Usage

You can also use the API endpoint:

```python
import requests

url = "http://127.0.0.1:5000/api/validate"
files = {'cv_file': open('your_cv.pdf', 'rb')}
response = requests.post(url, files=files)
results = response.json()
print(results)
```

## 📊 Scoring System

The system calculates an overall score (0-10) based on:

| Check | Weight | Description |
|-------|--------|-------------|
| Keywords | 10% | Technical skills detection |
| AI Analysis | 10% | HR manager perspective |
| Skills Separation | 8% | Technical vs Soft skills |
| GitHub | 8% | Repository links |
| Email | 8% | Professional email |
| Action Verbs | 7% | Impact words usage |
| Achievements | 7% | Quantifiable metrics |
| Contact Info | 6% | Complete contact details |
| Formatting | 6% | Visual quality |
| GPA | 5% | Academic performance |
| O/L & A/L | 5% | School results handling |
| Summary | 5% | Professional summary |
| Page Count | 5% | Appropriate length |
| Photo | 4% | Professional photo |
| Specialization | 4% | Area of expertise |

**Grades:**
- 8.5-10: Excellent ⭐⭐⭐
- 7.0-8.4: Good ⭐⭐
- 5.5-6.9: Fair ⭐
- <5.5: Needs Improvement ⚠️

## 🎨 Interface Features

### Main Upload Page
- Drag & drop file upload
- File size validation
- Real-time file info display
- Feature checklist

### Results Page
- Overall score card with grade
- Tabbed interface for easy navigation
- Detailed validation results with scores
- Job recommendations with match percentages
- Skills visualization
- Print-friendly format

## 🔍 Job Matching Algorithm

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity to match your CV with available jobs:

1. Extracts technical keywords from your CV
2. Compares with job descriptions
3. Calculates similarity scores
4. Ranks jobs by relevance

**Match Levels:**
- 🔥 **Excellent** (70%+): Highly recommended
- 👍 **Good** (50-70%): Worth applying
- ⭐ **Potential** (<50%): Consider if interested

## 📁 Project Structure

```
.
├── cv_validator_with_jobs.py  # Main application
├── config.py                  # Configuration (API keys)
├── requirements.txt           # Python dependencies
├── topjobs_it_jobs.csv       # Job listings database
├── templates/
│   ├── index.html            # Upload page
│   └── results.html          # Results page
└── uploads/                  # Temporary CV storage (auto-created)
```

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **PDF Processing**: PyMuPDF (pymupdf, pymupdf4llm)
- **AI Analysis**: Groq API (Llama 3.3 70B)
- **Job Matching**: scikit-learn (TF-IDF, Cosine Similarity)
- **Frontend**: HTML5, CSS3, JavaScript

## 📝 CV Best Practices (Sri Lankan IT Market)

Based on HR surveys and job listings analysis:

### ✅ DO:
- Keep it 1-2 pages
- Use professional email (firstname.lastname@domain.com)
- Include a professional photo
- Highlight technical skills clearly
- Mention projects with technologies used
- Add GitHub/LinkedIn profiles
- Use action verbs (developed, implemented, designed)
- Include quantifiable achievements (40% improvement, 5 projects)
- Add GPA if >3.0
- Include complete contact information

### ❌ DON'T:
- Include O/L & A/L results (not needed for IT jobs)
- Use unprofessional email addresses
- Exceed 2 pages for junior positions
- Use too many fonts or colors
- Put important info in headers/footers (ATS issues)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Developer

Janith - IT Manager & HND Cyber Security Student

---

Made with ❤️ for Sri Lankan IT Graduates 🇱🇰
