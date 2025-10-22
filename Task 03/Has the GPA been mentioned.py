import pymupdf  # PyMuPDF
import re
import time


start_time = time.time()

def check_gpa_in_cv(pdf_path):
    """
    Fast GPA checker - direct text extraction use කරනවා
    """
    try:
        # PDF open කරන්න
        doc = pymupdf.open(pdf_path)
        
        # සියලුම pages වල text extract කරන්න (super fast)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        
        # lowercase කරලා simple search එකක් (regex වලට වඩා වේගවත්)
        text_lower = full_text.lower()
        
        # GPA keywords check කරන්න
        if 'gpa' in text_lower or 'cgpa' in text_lower or 'grade point' in text_lower:
            print("✓ GPA එක mention කරලා තියෙනවා!")
            return True
        else:
            print("⚠ GPA එක mention කරලා නෑ!")
            print("📌 Suggestion: Add current GPA, if you have good results")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

# Usage
if __name__ == "__main__":
    pdf_file = r"C:\Users\Vikasitha\Downloads\Pawan Vikasitha.pdf"
    check_gpa_in_cv(pdf_file)

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.2f} seconds")