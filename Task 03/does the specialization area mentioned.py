import fitz  # PyMuPDF Library එකම තමා, මේක මේ නමිනුත් Import කරන්න පුළුවන්. කිසිම ප්‍රශ්නයක් නෑ
import re

def find_specialization_fast(pdf_path):
    
    # Specialize Area වල, එන්න පුළුවන් ඔක්කොම keywords මෙතන define කරන්න පුළුවන්.
    specializations = {
        "Software Technology": "software technology",
        "Network Technology": "network technology", 
        "Multimedia Technology": "multimedia technology"
    }
    
    try:
        # PDF open කරනවා
        doc = fitz.open(pdf_path)
        
        # මුලින් 3 pages check කරනවා.
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            # Page එකේ, text extract කරන්වා.
            page = doc[page_num]
            text = page.get_text().lower()
            
            # Specialization keywords check කරනවා, PDF එකේ text එකේ.
            for spec_name, keyword in specializations.items():
                if keyword in text:
                    doc.close()
                    return {
                        "found": True,
                        "specialization": spec_name,
                        "page": page_num + 1
                    }
        
        doc.close()
        return {"found": False, "specialization": None, "page": None}
        
    except Exception as e:
        return {"found": False, "error": str(e)}


pdf_path = r"C:\Users\Vikasitha\Documents\RP\Sample CVs\Hiruni_CV.pdf"
result = find_specialization_fast(pdf_path)

if result["found"]:
    print(f"✓ Specialization: {result['specialization']}")
    print(f"  (Page {result['page']} එකේ හම්බුණා)")
else:
    print("✗ Specialization area එකක් හම්බුණේ නැහැ")