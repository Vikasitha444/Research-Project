# pip install pymupdf4llm
import time
import pymupdf4llm


start_time = time.time()
markdown = pymupdf4llm.to_markdown(r"C:\Users\Vikasitha\Downloads\Pawan Vikasitha.pdf")

end_time = time.time()
execution_time = end_time - start_time

print(markdown)
print(f"\n\nTime taken: {execution_time:.4f} seconds")
