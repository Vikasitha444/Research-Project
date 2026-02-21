from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ---------- HOME ----------
@app.route("/")
def index():
    return render_template("index.html")

# ---------- CV ANALYZE ----------
@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    # your existing CV analyze logic here
    # example:
    return jsonify({"status": "CV analyzed successfully"})

# ---------- JOB RECOMMENDATIONS ----------
@app.route("/job-recommendations")
def job_recommendations():
    return render_template("job_recommendation.html")

if __name__ == "__main__":
    app.run(debug=True)
