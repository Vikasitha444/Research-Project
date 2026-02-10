# Job Recommendation System - Integration Guide

## Overview

This system integrates CV keyword extraction with a job recommendation engine that matches CV skills with Sri Lankan IT job opportunities using machine learning (TF-IDF and cosine similarity).

## Features

### 1. **CV Keyword Extraction**
- Automatically extracts technical keywords from uploaded CVs
- Identifies 35+ Sri Lankan tech market skills (Java, Python, React, AWS, etc.)
- Shows extracted keywords with visual tags

### 2. **Job Matching Algorithm**
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization
- Calculates **cosine similarity** between CV skills and job requirements
- Provides match scores (High: ≥70%, Medium: 50-70%, Low: <50%)

### 3. **Job Recommendations**
- 15 sample Sri Lankan IT jobs from major companies:
  - Dialog Axiata, IFS, Virtusa, WSO2, CodeGen, hSenid, Sysco LABS, etc.
- Real job attributes:
  - Job title, company, location, salary range
  - Job description with required skills
  - Application deadline and URL

### 4. **Skills Gap Analysis**
- Shows which skills you have
- Identifies missing skills to learn
- Calculates overall skill match percentage

### 5. **Market Insights**
- Total matching jobs available
- High/medium/low match distribution
- Average match score across all jobs

## How It Works

### User Flow:

```
1. Upload CV → 2. CV Analysis → 3. Click "Get Job Recommendations" → 4. View Matched Jobs
```

### Technical Flow:

```python
CV PDF → Keyword Extraction → Job Matcher → TF-IDF Vectorization → Cosine Similarity → Ranked Jobs
```

## File Structure

```
Task 05/
├── app.py                          # Flask routes (updated)
├── cv_validator_app.py            # CV validation logic
├── job_matcher.py                 # NEW: Job recommendation engine
├── test_integration.py            # NEW: Integration tests
├── templates/
│   ├── index.html                 # CV validator (updated with button)
│   └── job_recommendation.html    # NEW: Dynamic job display
└── .gitignore                     # NEW: Ignore Python cache
```

## Key Components

### 1. Job Matcher (`job_matcher.py`)

```python
class JobMatcher:
    def load_data()                           # Loads job data (CSV or sample)
    def get_recommendations(skills, top_n)    # Returns top N matching jobs
    def get_skills_gap_analysis()             # Analyzes skill gaps
    def get_market_insights()                 # Provides market statistics
```

**Algorithm:**
- Creates TF-IDF vectors for user skills + all job descriptions
- Computes cosine similarity between user vector and each job
- Ranks jobs by similarity score (0-100%)
- Returns top N recommendations

### 2. Flask Routes (`app.py`)

```python
@app.route("/job-recommendations")
def job_recommendations():
    # Gets keywords from URL or session
    # Calls JobMatcher
    # Returns rendered template with recommendations
```

**URL Example:**
```
http://localhost:5000/job-recommendations?keywords=Python+JavaScript+React+MongoDB
```

### 3. CV Validator Button (`index.html`)

After CV validation completes, a prominent button appears:

```html
<a href="/job-recommendations?keywords=Python JavaScript React...">
    💼 Get Job Recommendations
</a>
```

## Usage Instructions

### Running the Application

1. **Install Dependencies:**
   ```bash
   cd "Task 05"
   pip3 install pandas scikit-learn flask pymupdf openai
   ```

2. **Start the Flask Server:**
   ```bash
   python3 app.py
   ```

3. **Access the Application:**
   - Open browser: `http://localhost:5000`
   - Upload your CV
   - Wait for validation results
   - Click **"💼 Get Job Recommendations"** button

### Using Custom Job Data

To use your own job listings instead of sample data:

1. **Create CSV File:**
   ```
   Task 05/topjobs_it_jobs.csv
   ```

2. **Required Columns:**
   ```csv
   title,company,location,description,url,closing_date,salary_range
   ```

3. **Example Row:**
   ```csv
   "Software Engineer","Dialog Axiata","Colombo","Build applications with Python React Node.js MongoDB Docker","https://topjobs.lk/12345","2024-03-15","LKR 60,000 - 80,000"
   ```

4. **Restart Application:**
   The system will automatically load your CSV instead of sample data.

## Testing

Run the integration test:

```bash
cd "Task 05"
python3 test_integration.py
```

**Expected Output:**
```
✓ Job matcher initialized successfully
📋 Test Keywords: Python, JavaScript, React, Node.js, MongoDB...
✓ Found 5 job recommendations

Top 5 Job Matches:
1. Junior Full Stack Developer (Match: 26.9%)
2. Python Developer (Match: 20.6%)
...

✓ Skills You Have (6): python, javascript, react...
✗ Skills to Learn (5): docker, kubernetes, aws...
📊 Match Percentage: 54.5%

✅ All tests passed successfully!
```

## Match Score Interpretation

| Score Range | Level | Meaning |
|-------------|-------|---------|
| **70-100%** | 🔥 High | Excellent match - Apply immediately! |
| **50-69%**  | 👍 Medium | Good match - Worth considering |
| **0-49%**   | ⭐ Low | Potential match - May need skill development |

## Technical Details

### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=500,        # Top 500 most important words
    ngram_range=(1, 2),      # Single words + 2-word phrases
    stop_words='english'     # Ignore common words (the, a, is...)
)
```

### Cosine Similarity
```
similarity = (UserVector · JobVector) / (||UserVector|| × ||JobVector||)
```
- Returns value between 0 (no match) and 1 (perfect match)
- Converted to percentage (0-100%)

### Sample Job Database

The system includes 15 pre-loaded jobs:
- **Companies:** Dialog, IFS, Virtusa, WSO2, CodeGen, hSenid, Sysco LABS, 99X, Pearson, Axiata, Zone24x7, Fortude, Mobitel, Cambio, Informatics
- **Positions:** Full Stack, Frontend, Backend, Mobile, DevOps, QA, Cloud, Data Engineer
- **Locations:** Colombo (various areas), Colombo 02-08
- **Salary Range:** LKR 40,000 - 90,000
- **Types:** Internship, Trainee, Junior, Full-Time

## Future Enhancements

1. **Real-time Job Scraping:**
   - Integrate with TopJobs.lk API
   - Auto-update job database daily

2. **Advanced Filtering:**
   - Filter by location, salary, experience level
   - Sort by match score, date, salary

3. **Skills Visualization:**
   - Skill radar charts
   - Match heatmaps
   - Learning path suggestions

4. **Application Tracking:**
   - Save favorite jobs
   - Track application status
   - Set job alerts

5. **CV Optimization:**
   - Suggest CV improvements based on target job
   - Keyword gap analysis per job
   - ATS (Applicant Tracking System) optimization

## API Integration (Optional)

The system is ready for API integration. Example:

```python
# In app.py, add:
@app.route("/api/recommendations", methods=["POST"])
def api_recommendations():
    keywords = request.json.get('keywords', [])
    matcher = get_job_matcher()
    recommendations = matcher.get_recommendations(keywords, top_n=10)
    return jsonify(recommendations)
```

**Usage:**
```bash
curl -X POST http://localhost:5000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"keywords": ["Python", "React", "Node.js"]}'
```

## Troubleshooting

### Issue 1: No jobs displayed
**Solution:** Check if `topjobs_it_jobs.csv` exists or sample data is loading correctly.

### Issue 2: Low match scores
**Solution:** This is normal. TF-IDF works best with longer text. Real job descriptions will give better scores.

### Issue 3: Keywords not passing to job page
**Solution:** Ensure Flask sessions are enabled (`app.secret_key` is set).

### Issue 4: Import errors
**Solution:**
```bash
pip3 install pandas scikit-learn
```

## Support

For issues or questions:
1. Check test output: `python3 test_integration.py`
2. Review Flask logs in terminal
3. Verify all files are in correct locations

## License

This project is part of the CV Validator Pro research project.

---

**Created:** 2024-02-10
**Version:** 1.0
**Dependencies:** Flask, pandas, scikit-learn, pymupdf
