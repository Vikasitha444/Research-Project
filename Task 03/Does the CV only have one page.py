import pymupdf

def check_cv_page_count(pdf_path):
    """
    CV එකේ pages ගණන fast check කරන්න
    """
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)  # Super fast!
        doc.close()
        
        if page_count > 1:
            print(f"⚠ CV එකේ pages {page_count}ක් තියෙනවා!")
            print(" උපරිම තිබිය හැක්කේ, එක page එකක් පමණයි.")
            return False
        else:
            print(f"✓ CV එක page {page_count}කින් තියෙනවා - Perfect!")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

# Usage
if __name__ == "__main__":
    pdf_file = r"C:\Users\Vikasitha\Documents\RP\Sample CVs\RAVINDU LAKSHAN.pdf"
    check_cv_page_count(pdf_file)