import csv
import sys
import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import io
import sys


CSV_PATH = "IT_Job_Roles_Skills.csv"
PICKLE_PATH = "job_matcher.pkl"


buffer = io.StringIO()
sys.stdout = buffer  # Console redirect කරනවා buffer එකට



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



df, vectorizer, X, le = build_pipeline(CSV_PATH)
models = load_models(PICKLE_PATH)

sample_skills = "Python, Machine Learing, Deep Learning, JavaScrit, Figma"
print(f"\nPredicting job roles for skills: {sample_skills}")
predicted = predict_job(sample_skills, df, vectorizer, X, le, models)
print(f"\nFinal recommendation: {predicted}\n")
print("Preitced again :::::: ",predicted)










def extract_all_keywords(special_text):

    lines = special_text.strip().splitlines()

    best_match_found = False
    keywords = []
    seen = set()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # "BEST MATCH" line හොයන්න
        if "BEST MATCH" in stripped.upper():
            best_match_found = True

            # එකම line එකේ job title තියෙනවාද — BEST MATCH + === ඉවර වෙලා
            after_best_match = re.split(r'BEST MATCH.*?(?:={3,})', stripped, flags=re.IGNORECASE)
            if len(after_best_match) > 1:
                remainder = after_best_match[-1].strip()
                cleaned = _clean_line(remainder)
                if cleaned and cleaned.upper() not in seen:
                    keywords.append(cleaned)
                    seen.add(cleaned.upper())
            continue

        # BEST MATCH හොයාගත්තට පස්සේ, ඕනෑම non-empty line
        if best_match_found:
            cleaned = _clean_line(stripped)
            if cleaned and cleaned.upper() not in seen:
                keywords.append(cleaned)
                seen.add(cleaned.upper())

    return keywords


def _clean_line(text):
    """
    Line එකෙන් prefix symbols (⭐, =, -, *, #) සහ (X votes) remove කරලා
    job title clean කරන function.
    """
    # ඉස්සරහින් ⭐ සහ non-letter symbols strip කරන්න
    cleaned = re.sub(r'^[\W⭐]+', '', text, flags=re.UNICODE).strip()
    # "(X votes)" remove කරන්න
    cleaned = re.sub(r'\(\d+\s*votes?\)', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def search_in_csv(file_path, search_text):
    """
    CSV File එකේ Text එකක් හොයන function.

    Args:
        file_path: CSV file path
        search_text: හොයන text
        case_sensitive: True = exact case, False = case ignore
    """

    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' file හොයාගන්න බැරිවුණා!")
        return

    results = []

    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames

            if not headers:
                print("❌ CSV file එකේ headers නෑ!")
                return

            print(f"\n📂 File: {file_path}")
            print(f"🔍 සොයන text: '{search_text}'")
            print(f"📋 Columns: {', '.join(headers)}")
            print("-" * 60)

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                for col_name, cell_value in row.items():
                    if cell_value is None:
                        continue

                    found = search_text.lower() in cell_value.lower()

                    if found:
                        results.append({
                            'row': row_num,
                            'column': col_name,
                            'value': cell_value,
                            'full_row': row
                        })
                        break  # එකම row එකේ multiple matches avoid කරන්න

    except UnicodeDecodeError:
        # UTF-8 fail වුණොත් සිංහල encoding try කරන්න
        with open(file_path, newline='', encoding='latin-1') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames

            for row_num, row in enumerate(reader, start=2):
                for col_name, cell_value in row.items():
                    if cell_value is None:
                        continue
                    found = search_text.lower() in cell_value.lower()

                    if found:
                        results.append({
                            'row': row_num,
                            'column': col_name,
                            'value': cell_value,
                            'full_row': row
                        })
                        break

    # Results print කරන්න
    if results:
        print(f"✅ {len(results)} result(s) හොයාගත්තා!\n")
        for idx, result in enumerate(results, 1):
            print(f"🎯 Result {idx}:")
            print(f"   Row Number : {result['row']}")
            print(f"   Column     : {result['column']}")
            print(f"   Found in   : {result['value']}")
            print(f"   Full Row   :")
            for col, val in result['full_row'].items():
                print(f"      {col}: {val}")
            print("-" * 60)
    else:
        print(f"❌ '{search_text}' CSV file එකේ හොයාගන්න බැරිවුණා.")

    return results




CSV_FILE = "D:\Previous Document Folder\RP\Task 07\Scraped Jobs CSV files\Intern only.csv"  # CSV file path

def extract_best_match_section(output_text):
    # Find the section starting from the separator line before BEST MATCH
    pattern = r'={10,}\s*\n.*?🏆 BEST MATCH.*?(?=\n✅|\Z)'
    match = re.search(pattern, output_text, re.DOTALL)

    if match:
        return match.group(0).strip()
    return None

sys.stdout = sys.__stdout__  # Console normal කරනවා
captured = buffer.getvalue()  # Variable එකට

print("Captured::::::::::::::::::::::::::::::::::::::::::::::::::", captured,"::::::::::::::::::::::::::::::::::::::::::::::::::")

output = captured



result = extract_best_match_section(output)
print(result)

SPECIAL_TEXT = result

# ====================================================


def main():
    print("=" * 60)
    print("       CSV File Search Tool - Python")
    print("=" * 60)

    # Special text එකෙන් සියලු keywords extract කරන්න
    keywords = extract_all_keywords(SPECIAL_TEXT)

    if not keywords:
        print("❌ SPECIAL_TEXT එකෙන් keywords extract කරගන්න බැරිවුණා!")
        return

    print(f"\n📌 Extract කළ Keywords ({len(keywords)}):")
    for i, kw in enumerate(keywords, 1):
        print(f"   {i}. {kw}")

    print("\n" + "=" * 60)
    print("       Search Results")
    print("=" * 60)

    # සියලු keywords loop කරලා search කරන්න
    all_results = {}
    for keyword in keywords:
        print(f"\n🔎 Searching: '{keyword}'")
        print("=" * 60)
        results = search_in_csv(CSV_FILE, keyword)
        all_results[keyword] = results if results else []

    # Summary
    print("\n" + "=" * 60)
    print("       Summary")
    print("=" * 60)
    for keyword, results in all_results.items():
        count = len(results) if results else 0
        status = f"✅ {count} match(es)" if count > 0 else "❌ No matches"
        print(f"  '{keyword}' → {status}")


if __name__ == "__main__":
    main()