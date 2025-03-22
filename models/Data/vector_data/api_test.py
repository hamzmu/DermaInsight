import requests
import json

# File path (use raw string or forward slashes)
pdf_file_path = r"C:\Users\Kaushal\Desktop\GenAI\models\Data\vector_data\ClinicalDermatology.pdf"

# Endpoint URLs
ingest_pdf_url = "http://127.0.0.1:8000/ingest_pdf/"
diagnose_url = "http://127.0.0.1:8000/diagnose/"

def test_ingest_pdf():
    print("Testing PDF ingestion...")
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            files = {"file": ("ClinicalDermatology.pdf", pdf_file, "application/pdf")}
            ingest_response = requests.post(ingest_pdf_url, files=files)

        if ingest_response.status_code == 200:
            print("PDF ingested successfully!")
            print(f"Response: {ingest_response.json()}")
            return True
        else:
            print(f"Failed to ingest PDF: {ingest_response.status_code}")
            print(f"Error message: {ingest_response.text}")
            return False
    except Exception as e:
        print(f"Exception during PDF ingestion: {e}")
        return False

def test_diagnosis():
    print("\nTesting diagnosis with symptoms...")
    try:
        # The key issue was here - needs to be sent as JSON
        diagnosis_data = {"symptoms": "Patient shows symptoms of severe itching, dry patches on elbows."}
        headers = {"Content-Type": "application/json"}
        
        diagnosis_response = requests.post(
            diagnose_url, 
            data=json.dumps(diagnosis_data),
            headers=headers
        )

        if diagnosis_response.status_code == 200:
            print("Diagnosis response received!")
            print(f"Diagnosis: {diagnosis_response.json()}")
            return True
        else:
            print(f"Failed to diagnose: {diagnosis_response.status_code}")
            print(f"Error message: {diagnosis_response.text}")
            return False
    except Exception as e:
        print(f"Exception during diagnosis: {e}")
        return False

if __name__ == "__main__":
    # First ingest PDF, then test diagnosis
    if test_ingest_pdf():
        print("\nPDF ingestion successful, proceeding to diagnosis test...")
        test_diagnosis()
    else:
        print("\nSkipping diagnosis test because PDF ingestion failed.")