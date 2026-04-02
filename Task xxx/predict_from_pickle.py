import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


CSV_PATH = "IT_Job_Roles_Skills.csv"
PICKLE_PATH = "job_matcher.pkl"


def build_pipeline(csv_path: str):
    df = pd.read_csv(csv_path, encoding='latin1')
    # Ensure Skills column is string
    skills = df["Skills"].astype(str)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(skills)

    le = LabelEncoder()
    y = df["Job Title"].astype(str)
    le.fit(y)

    return df, vectorizer, X, le


def load_models(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        models = pickle.load(f)
    return models


def predict_job(input_skills: str, df, vectorizer, X, le, models, top_n: int = 3):
    input_vec = vectorizer.transform([input_skills])

    # Cosine similarity baseline
    similarities = cosine_similarity(input_vec, X)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    print("\n📌 METHOD 1: TF-IDF Cosine Similarity")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {df['Job Title'].iloc[idx]} ({similarities[idx]*100:.1f}%)")

    # ML model predictions
    print("\n📌 METHOD 2: ML Model Predictions")
    model_results = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_vec)[0]
            # model.classes_ contains encoded class labels
            top_idx = proba.argsort()[::-1][:top_n]
            results = []
            for i in top_idx:
                class_label = model.classes_[i]
                title = le.inverse_transform([int(class_label)])[0]
                results.append((title, proba[i] * 100))
        else:
            pred = model.predict(input_vec)[0]
            title = le.inverse_transform([int(pred)])[0]
            results = [(title, 100.0)]

        model_results[name] = results
        print(f"\n  {name}:")
        for title, score in results:
            print(f"     → {title} ({score:.1f}%)")

    # Voting
    votes = {}
    for name, results in model_results.items():
        top_title = results[0][0]
        votes[top_title] = votes.get(top_title, 0) + 1

    cos_top = df['Job Title'].iloc[top_indices[0]]
    votes[cos_top] = votes.get(cos_top, 0) + 1

    best = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("🏆 BEST MATCH (Majority Vote)")
    for title, vote_count in best:
        print(f"  {'⭐' * vote_count} {title} ({vote_count} votes)")

    print(f"\n✅ RECOMMENDED: {best[0][0]}")
    return best[0][0]


def main():
    df, vectorizer, X, le = build_pipeline(CSV_PATH)
    models = load_models(PICKLE_PATH)

    sample_skills = "Leadership"
    print(f"\nPredicting job roles for skills: {sample_skills}")
    predicted = predict_job(sample_skills, df, vectorizer, X, le, models)
    print(f"\nFinal recommendation: {predicted}\n")


if __name__ == '__main__':
    main()
