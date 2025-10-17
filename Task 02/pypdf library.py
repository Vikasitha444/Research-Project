import time
from pypdf import PdfReader


start_time = time.time()

reader = PdfReader(r"C:\Users\Vikasitha\Downloads\Pawan Vikasitha.pdf")
text = "\n".join(p.extract_text() for p in reader.pages)

end_time = time.time()
execution_time = end_time - start_time

print(text)
print(f"\n\nTime taken: {execution_time:.4f} seconds")