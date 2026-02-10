from flask import Flask, render_template, request, session, redirect, url_for
from job_matcher import get_job_matcher

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/job-recommendations")
def job_recommendations():
    """
    Job recommendations page
    Accepts keywords from URL parameter or session
    """
    # Get keywords from URL parameter or session
    keywords = request.args.get('keywords', '')

    if not keywords and 'cv_keywords' in session:
        keywords = session.get('cv_keywords', '')

    # Store in session for future use
    if keywords:
        session['cv_keywords'] = keywords

    # Get location filter
    location_filter = request.args.get('location', 'All')

    # Get number of recommendations
    num_jobs = int(request.args.get('num_jobs', 10))

    # Initialize job matcher
    matcher = get_job_matcher()

    # Get recommendations if keywords provided
    if keywords:
        recommendations = matcher.get_recommendations(keywords, location_filter, num_jobs)
        insights = matcher.get_market_insights(keywords)
    else:
        # Show sample jobs if no keywords
        recommendations = matcher.get_recommendations("Python JavaScript React", top_n=10)
        insights = None

    # Get available locations from data
    locations = ['All'] + sorted(list(set([job['location'] for job in recommendations])))

    # Calculate statistics
    high_match = sum(1 for r in recommendations if r['match_level'] == 'high')
    medium_match = sum(1 for r in recommendations if r['match_level'] == 'medium')

    return render_template(
        'job_recommendation.html',
        recommendations=recommendations,
        keywords=keywords,
        location_filter=location_filter,
        locations=locations,
        insights=insights,
        stats={
            'total_jobs': len(recommendations),
            'high_match': high_match,
            'medium_match': medium_match,
            'low_match': len(recommendations) - high_match - medium_match
        }
    )

@app.route("/set-keywords", methods=["POST"])
def set_keywords():
    """
    API endpoint to set keywords from CV validation
    """
    data = request.get_json()
    keywords = data.get('keywords', [])

    if isinstance(keywords, list):
        keywords_str = ' '.join(keywords)
    else:
        keywords_str = keywords

    session['cv_keywords'] = keywords_str

    return {'success': True, 'keywords': keywords_str}

if __name__ == "__main__":
    app.run(debug=True)
