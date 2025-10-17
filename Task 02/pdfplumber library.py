import time
import pdfplumber

start_time = time.time()

with pdfplumber.open(r"C:\Users\Vikasitha\Downloads\Pawan Vikasitha.pdf") as pdf:
    first_page = pdf.pages[0]
    text = first_page.extract_text()
    table = first_page.extract_table()

    end_time = time.time()
    execution_time = end_time - start_time

    print(text)
    print(f"\n\nTime taken: {execution_time:.4f} seconds")