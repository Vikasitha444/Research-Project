"""
Job Recommendation Engine
Matches CV keywords with job requirements using TF-IDF and cosine similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class JobMatcher:
    def __init__(self, csv_path='topjobs_it_jobs.csv'):
        """Initialize the job matcher with job data"""
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load and prepare job data"""
        if not os.path.exists(self.csv_path):
            # Create sample data if CSV doesn't exist
            self.df = self.create_sample_jobs()
        else:
            self.df = pd.read_csv(self.csv_path)
            self.prepare_data()

    def prepare_data(self):
        """Clean and prepare job data"""
        self.df['title'] = self.df['title'].fillna('Not specified')
        self.df['company'] = self.df['company'].fillna('Company Name Withheld')
        self.df['location'] = self.df['location'].fillna('Sri Lanka')
        self.df['description'] = self.df['description'].fillna('Please refer the vacancy')
        self.df['combined_text'] = self.df['title'].astype(str) + ' ' + self.df['description'].astype(str)

    def create_sample_jobs(self):
        """Create sample job data for testing"""
        jobs = [
            {
                'title': 'Junior Full Stack Developer',
                'company': 'Dialog Axiata PLC',
                'location': 'Colombo',
                'description': 'Looking for a talented developer with React Node.js MongoDB experience. Build modern web applications using JavaScript Python MySQL Git Docker.',
                'url': 'https://topjobs.lk/job/12345',
                'closing_date': '2024-03-15',
                'salary_range': 'LKR 60,000 - 80,000'
            },
            {
                'title': 'Software Engineer Intern',
                'company': 'IFS - Sri Lanka',
                'location': 'Colombo 03',
                'description': 'Entry-level position for graduates. Work with Java Python SQL Spring Boot Docker Kubernetes. Learn microservices architecture REST API development.',
                'url': 'https://topjobs.lk/job/12346',
                'closing_date': '2024-03-20',
                'salary_range': 'LKR 45,000 - 55,000'
            },
            {
                'title': 'Frontend Developer',
                'company': 'Virtusa Corporation',
                'location': 'Colombo 07',
                'description': 'Create stunning user interfaces with React JavaScript TypeScript HTML5 CSS3. Experience with Angular Vue.js is a plus. Work on enterprise applications.',
                'url': 'https://topjobs.lk/job/12347',
                'closing_date': '2024-03-18',
                'salary_range': 'LKR 55,000 - 75,000'
            },
            {
                'title': 'Backend Developer Trainee',
                'company': 'WSO2 (Private) Limited',
                'location': 'Colombo 05',
                'description': 'Join our backend team. Work with Java Node.js MySQL MongoDB. Build REST API GraphQL microservices. Learn Agile Scrum methodologies.',
                'url': 'https://topjobs.lk/job/12348',
                'closing_date': '2024-03-22',
                'salary_range': 'LKR 50,000 - 65,000'
            },
            {
                'title': 'UI/UX Developer',
                'company': 'CodeGen International',
                'location': 'Colombo 02',
                'description': 'Design and develop user interfaces. Skills needed: HTML5 CSS3 JavaScript React Figma Adobe XD. Create responsive web designs.',
                'url': 'https://topjobs.lk/job/12349',
                'closing_date': '2024-03-25',
                'salary_range': 'LKR 50,000 - 70,000'
            },
            {
                'title': 'Mobile App Developer Intern',
                'company': 'hSenid Mobile Solutions',
                'location': 'Colombo 08',
                'description': 'Build mobile applications using React Native Flutter Android iOS. JavaScript TypeScript experience required. Work on cutting-edge mobile projects.',
                'url': 'https://topjobs.lk/job/12350',
                'closing_date': '2024-03-19',
                'salary_range': 'LKR 40,000 - 50,000'
            },
            {
                'title': 'DevOps Engineer Trainee',
                'company': 'Sysco LABS',
                'location': 'Colombo',
                'description': 'Learn DevOps practices. Work with Docker Kubernetes AWS Azure Jenkins Git GitHub. Automate deployment pipelines and cloud infrastructure.',
                'url': 'https://topjobs.lk/job/12351',
                'closing_date': '2024-03-28',
                'salary_range': 'LKR 55,000 - 70,000'
            },
            {
                'title': 'Java Developer',
                'company': '99X Technology',
                'location': 'Colombo 03',
                'description': 'Develop enterprise applications using Java Spring Boot MySQL PostgreSQL. Experience with Microservices REST API Agile methodologies required.',
                'url': 'https://topjobs.lk/job/12352',
                'closing_date': '2024-03-30',
                'salary_range': 'LKR 65,000 - 85,000'
            },
            {
                'title': 'Python Developer',
                'company': 'Pearson Lanka',
                'location': 'Colombo 05',
                'description': 'Build scalable applications with Python Django Flask. Work with PostgreSQL MongoDB Redis. Experience with REST API Docker is preferred.',
                'url': 'https://topjobs.lk/job/12353',
                'closing_date': '2024-04-02',
                'salary_range': 'LKR 60,000 - 80,000'
            },
            {
                'title': 'Full Stack JavaScript Developer',
                'company': 'Axiata Digital Labs',
                'location': 'Colombo',
                'description': 'Work with full JavaScript stack: React Angular Vue.js Node.js Express MongoDB. Build modern web applications using Agile JIRA Git.',
                'url': 'https://topjobs.lk/job/12354',
                'closing_date': '2024-04-05',
                'salary_range': 'LKR 70,000 - 90,000'
            },
            {
                'title': 'QA Engineer Intern',
                'company': 'Zone24x7',
                'location': 'Colombo 07',
                'description': 'Learn software testing methodologies. Manual and automated testing with Selenium. Basic programming knowledge in Java Python JavaScript required.',
                'url': 'https://topjobs.lk/job/12355',
                'closing_date': '2024-03-16',
                'salary_range': 'LKR 40,000 - 55,000'
            },
            {
                'title': 'React Developer',
                'company': 'Fortude (Pvt) Ltd',
                'location': 'Colombo',
                'description': 'Expert in React JavaScript TypeScript HTML5 CSS3. Build component libraries and SPAs. Redux state management experience is essential.',
                'url': 'https://topjobs.lk/job/12356',
                'closing_date': '2024-03-21',
                'salary_range': 'LKR 65,000 - 85,000'
            },
            {
                'title': 'Android Developer',
                'company': 'Mobitel (Pvt) Ltd',
                'location': 'Colombo 02',
                'description': 'Develop native Android applications using Java Kotlin. Experience with Firebase REST API Git. Knowledge of Flutter is a plus.',
                'url': 'https://topjobs.lk/job/12357',
                'closing_date': '2024-04-08',
                'salary_range': 'LKR 60,000 - 80,000'
            },
            {
                'title': 'Cloud Engineer Trainee',
                'company': 'Cambio Software Engineering',
                'location': 'Colombo',
                'description': 'Learn cloud technologies AWS Azure Google Cloud. Work with Docker Kubernetes Terraform. Basic Linux Python knowledge required.',
                'url': 'https://topjobs.lk/job/12358',
                'closing_date': '2024-04-10',
                'salary_range': 'LKR 50,000 - 70,000'
            },
            {
                'title': 'Data Engineer Intern',
                'company': 'Informatics Institute',
                'location': 'Colombo 06',
                'description': 'Work with data pipelines ETL processes. Skills: Python SQL MySQL PostgreSQL MongoDB. Experience with data visualization tools preferred.',
                'url': 'https://topjobs.lk/job/12359',
                'closing_date': '2024-04-12',
                'salary_range': 'LKR 45,000 - 60,000'
            }
        ]

        df = pd.DataFrame(jobs)
        df['combined_text'] = df['title'] + ' ' + df['description']
        return df

    def get_recommendations(self, user_skills, location_filter='All', top_n=10):
        """
        Get job recommendations based on user skills

        Args:
            user_skills (list or str): List of skills or comma-separated string
            location_filter (str): Location to filter jobs
            top_n (int): Number of recommendations to return

        Returns:
            list: List of recommended jobs with match scores
        """
        if isinstance(user_skills, list):
            user_skills_text = ' '.join(user_skills)
        else:
            user_skills_text = user_skills

        # Filter by location if specified
        df_filtered = self.df.copy()
        if location_filter != 'All' and location_filter:
            df_filtered = df_filtered[df_filtered['location'].str.contains(location_filter, case=False, na=False)]

        if len(df_filtered) == 0:
            return []

        # Create TF-IDF vectors
        all_text = list(df_filtered['combined_text']) + [user_skills_text]
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_text)

        # Calculate similarity
        user_vector = tfidf_matrix[-1]
        job_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(user_vector, job_vectors).flatten()

        # Add similarity scores
        df_filtered = df_filtered.copy()
        df_filtered['similarity_score'] = similarities

        # Sort and get top N
        recommendations = df_filtered.sort_values('similarity_score', ascending=False).head(top_n)

        # Convert to list of dicts for easier template rendering
        results = []
        for _, row in recommendations.iterrows():
            match_percentage = row['similarity_score'] * 100

            # Determine match level
            if match_percentage >= 70:
                match_level = 'high'
                match_text = 'Excellent'
            elif match_percentage >= 50:
                match_level = 'medium'
                match_text = 'Good'
            else:
                match_level = 'low'
                match_text = 'Potential'

            results.append({
                'title': row['title'],
                'company': row['company'],
                'location': row['location'],
                'description': row['description'],
                'url': row.get('url', '#'),
                'closing_date': row.get('closing_date', 'N/A'),
                'salary_range': row.get('salary_range', 'Not specified'),
                'match_score': round(match_percentage, 1),
                'match_level': match_level,
                'match_text': match_text
            })

        return results

    def get_skills_gap_analysis(self, user_skills, job_requirements):
        """
        Analyze skills gap between user and job market

        Args:
            user_skills (list): User's current skills
            job_requirements (list): Skills required in jobs

        Returns:
            dict: Skills gap analysis
        """
        user_skills_set = set([s.lower() for s in user_skills])
        job_skills_set = set([s.lower() for s in job_requirements])

        has_skills = user_skills_set & job_skills_set
        missing_skills = job_skills_set - user_skills_set

        return {
            'has_skills': list(has_skills),
            'missing_skills': list(missing_skills),
            'total_match': len(has_skills),
            'total_gap': len(missing_skills),
            'match_percentage': (len(has_skills) / len(job_skills_set) * 100) if job_skills_set else 0
        }

    def get_market_insights(self, user_skills):
        """
        Get insights about job market based on user skills

        Returns:
            dict: Market insights
        """
        # Get all recommendations
        recommendations = self.get_recommendations(user_skills, top_n=100)

        if not recommendations:
            return None

        # Calculate statistics
        high_match = sum(1 for r in recommendations if r['match_level'] == 'high')
        medium_match = sum(1 for r in recommendations if r['match_level'] == 'medium')

        return {
            'total_jobs': len(recommendations),
            'high_match_count': high_match,
            'medium_match_count': medium_match,
            'average_match': sum(r['match_score'] for r in recommendations) / len(recommendations),
            'top_companies': list(set([r['company'] for r in recommendations[:10]])),
            'top_locations': list(set([r['location'] for r in recommendations[:10]]))
        }

# Singleton instance
_job_matcher_instance = None

def get_job_matcher():
    """Get or create job matcher instance"""
    global _job_matcher_instance
    if _job_matcher_instance is None:
        _job_matcher_instance = JobMatcher()
    return _job_matcher_instance
