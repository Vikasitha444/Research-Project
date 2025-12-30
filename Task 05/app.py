from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/job-recommendations")
def job_recommendations():
    return render_template("job_recommendation.html")

if __name__ == "__main__":
    app.run(debug=True)
