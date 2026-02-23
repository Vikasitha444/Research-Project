import csv
import sys
import os
import re

def extract_all_keywords(special_text):
    """
    Special text එකෙන් BEST MATCH section එකේ සියලු job titles extract කරන function.
    ඉස්සරහින් ⭐, =, හෝ වෙනත් symbols තිබ්බත් හරියට extract කරනවා.
    Duplicate titles (case-insensitive) ignore කරනවා.
    """
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


# ====================================================
# ✏️  මෙතන වෙනස් කරන්න
# ====================================================

CSV_FILE = "D:\Previous Document Folder\RP\Task 07\Scraped Jobs CSV files\Intern only.csv"  # CSV file path

SPECIAL_TEXT = """
============================================================ 🏆 BEST MATCH (Majority Vote) ============================================================   ⭐⭐⭐ Director of Engineering (3 votes)
⭐⭐ TECHNICAL LEAD (2 votes)
⭐ BUSINESS SYSTEMS ANALYST (1 votes)
⭐ DIRECTOR OF ENGINEERING (1 votes)
Des
"""

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