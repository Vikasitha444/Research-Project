import os

file_path = input("CV file path: ").strip().strip('"').strip("'")
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        pass
else:
    print("File Location is incorrect")