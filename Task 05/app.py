from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'cv_file' not in request.files:
            return render_template("index.html", 
                                 results=None, 
                                 error="කරුණාකර CV file එකක් select කරන්න")
        
        file = request.files['cv_file']
        
        # Check if filename is empty
        if file.filename == '':
            return render_template("index.html", 
                                 results=None, 
                                 error="කරුණාකර CV file එකක් select කරන්න")
        
        # Check if file is PDF
        if not allowed_file(file.filename):
            return render_template("index.html", 
                                 results=None, 
                                 error="PDF files පමණක් accept කරනවා")
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Mock results for demonstration
            # ඔබට මෙතන ඔයාගේ cv_validator_app.py එකේ validation logic එක call කරන්න පුළුවන්
            results = {
                'overall': {
                    'score': 8.5,
                    'grade': 'Excellent',
                    'status': 'success',
                    'message': 'CV validation complete!'
                },
                'page_count': {
                    'status': 'success',
                    'message': 'Perfect! Single page CV.',
                    'value': '1 page ✓',
                    'score': 10
                },
                'gpa': {
                    'status': 'success',
                    'message': 'GPA is mentioned in the CV.',
                    'value': 'Found ✓',
                    'score': 10
                },
                'professional_email': {
                    'status': 'success',
                    'message': 'Email address looks professional.',
                    'value': 'Professional ✓',
                    'score': 10
                },
                'photo': {
                    'status': 'success',
                    'message': 'Professional photo found.',
                    'value': 'Present ✓',
                    'score': 10
                },
                'ol_al': {
                    'status': 'success',
                    'message': 'No school exam results found. Good!',
                    'value': 'Not present ✓',
                    'score': 10
                },
                'formatting': {
                    'status': 'success',
                    'message': 'Formatting looks good.',
                    'value': 'Good ✓',
                    'score': 10
                },
                'specialization': {
                    'status': 'success',
                    'message': 'Specialization: Software Technology',
                    'value': 'Software Technology',
                    'score': 10
                },
                'github': {
                    'status': 'success',
                    'message': 'Found 2 links, 2 are working.',
                    'value': '2/2 working',
                    'repos': [
                        {'url': 'https://github.com/example/project1', 'valid': True, 'message': 'Working'},
                        {'url': 'https://github.com/example/project2', 'valid': True, 'message': 'Working'}
                    ],
                    'score': 10
                },
                'skills': {
                    'status': 'success',
                    'message': 'Both Technical and Soft Skills sections found.',
                    'value': 'Both present ✓',
                    'score': 10,
                    'soft_count': 5,
                    'tech_count': 8
                },
                'keywords': {
                    'status': 'success',
                    'message': 'Excellent! Found 12 relevant technical keywords.',
                    'value': '12 keywords',
                    'found_keywords': ['Python', 'JavaScript', 'React', 'Node.js', 'Git', 'MySQL', 'HTML5', 'CSS3', 'Docker', 'AWS'],
                    'score': 10
                },
                'contact': {
                    'status': 'success',
                    'message': 'Complete contact information found.',
                    'value': 'Complete ✓',
                    'score': 10,
                    'details': ['Phone', 'Email', 'LinkedIn', 'Location']
                },
                'action_verbs': {
                    'status': 'success',
                    'message': 'Excellent! Found 10 strong action verbs.',
                    'value': '10 verbs ✓',
                    'score': 10,
                    'verbs': ['developed', 'designed', 'implemented', 'created', 'built', 'optimized', 'delivered', 'collaborated']
                },
                'achievements': {
                    'status': 'success',
                    'message': 'Excellent! Found 6 quantifiable achievements.',
                    'value': '6 metrics ✓',
                    'score': 10
                },
                'summary': {
                    'status': 'success',
                    'message': 'Professional summary found at top of CV.',
                    'value': 'Present ✓',
                    'score': 10
                },
                'llm': {
                    'status': 'success',
                    'message': 'AI Analysis complete',
                    'results': [
                        {'name': 'Professional Photo', 'description': 'Photo present', 'passed': True},
                        {'name': 'Professional Email', 'description': 'Email is professional', 'passed': True},
                        {'name': 'Technical Skills', 'description': 'Skills clearly shown', 'passed': True},
                        {'name': 'Projects', 'description': 'Technologies mentioned', 'passed': True},
                        {'name': 'Formatting', 'description': 'Well formatted', 'passed': True},
                        {'name': 'School Results', 'description': 'Appropriately handled', 'passed': True},
                        {'name': 'Length', 'description': '1-2 pages', 'passed': True},
                        {'name': 'Keywords', 'description': 'Technical keywords present', 'passed': True},
                        {'name': 'Degree', 'description': 'ICT degree mentioned', 'passed': True},
                        {'name': 'References', 'description': 'References included', 'passed': True}
                    ],
                    'passed': 10,
                    'total': 10,
                    'score': 10
                }
            }
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return render_template("index.html", results=results, filename=filename)
    
    # GET request - show upload form
    return render_template("index.html", results=None)

@app.route("/job-recommendations")
def job_recommendations():
    return render_template("job_recommendation.html")

@app.route("/jobrecommendation.html")
def job_recommendations_alt():
    return render_template("job_recommendation.html")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CV Validator Pro - Working Version")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)