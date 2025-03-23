import requests
import json

# Define the URL of the RAG API
RAG_API_URL = "http://127.0.0.1:8001/diagnose/"

# Define the input data
input_data = {
    "vit_analysis": {
        "acne": {"class": "Level 1", "confidence": 0.95},
        "disease": {"class": "Psoriasis", "confidence": 0.85},
        "type": {"class": "Oily", "confidence": 0.90},
        "cancer": {"class": "Benign", "confidence": 0.98}
    },
    "user_input": "I have red, itchy patches on my skin.",
    "combined_query": "Image analysis: {...}. User description: I have red, itchy patches on my skin."
}

# Send the POST request to the RAG API
try:
    response = requests.post(
        RAG_API_URL,
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()  # Raise an error for bad status codes

    # Print the response
    print("Response from RAG API:")
    print(json.dumps(response.json(), indent=4))

except requests.exceptions.RequestException as e:
    print(f"Error calling RAG API: {e}")