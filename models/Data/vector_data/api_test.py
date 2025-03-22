import os
import requests

def ingest_pdfs_from_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for pdf in pdf_files:
        pdf_path = os.path.join(folder_path, pdf)
        response = requests.post("http://localhost:8000/ingest/", json={"pdf_path": pdf_path})
        print(f"Ingested {pdf}: {response.json()}")

# Set your folder path here
folder_path = "path/to/your/folder"
ingest_pdfs_from_folder(folder_path)