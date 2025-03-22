from fastapi import FastAPI, HTTPException, UploadFile, File
import os
from dotenv import load_dotenv
import uuid
import re
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import fitz  # PyMuPDF for extracting text from PDFs

# Load environment variables from .env file
load_dotenv()

# Ensure API keys are set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise EnvironmentError("API keys for Pinecone and/or Gemini are not set in the environment variables.")

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone client with a valid region (adjust region if needed)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dermatology-chunks"
available_indexes = pc.list_indexes().names()

if index_name not in available_indexes:
    try:
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust to your embedding model's dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Use a supported region for your plan
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating index: {e}")

index = pc.Index(index_name)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load the reranker model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        # Load the PDF from bytes using fitz (PyMuPDF)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {e}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-(overlap // 2):]  # retain some overlap
            current_length = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embedding(text: str) -> list:
    # Call the Gemini API to get embeddings
    response = genai.embed_content(model="text-embedding-gecko", content=text)
    return response["embedding"]


def store_in_pinecone(text_chunks: list):
    vectors = [
        (str(uuid.uuid4()), get_embedding(chunk), {"text": chunk})
        for chunk in text_chunks
    ]
    try:
        index.upsert(vectors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing vectors in Pinecone: {e}")


@app.post("/ingest_pdf/")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Endpoint to ingest a PDF file.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

    text_chunks = chunk_text(text)
    store_in_pinecone(text_chunks)
    return {"message": "PDF ingested successfully", "num_chunks": len(text_chunks)}


def search_pinecone(query_text: str, top_k: int = 5) -> list:
    query_embedding = get_embedding(query_text)
    results = index.query(queries=[query_embedding], top_k=top_k, include_metadata=True)
    return [match.metadata["text"] for match in results["matches"]]


def rerank_chunks(query: str, chunks: list) -> list:
    # Combine query and chunks for reranking
    inputs = tokenizer([query] + chunks, padding=True, truncation=True, return_tensors="pt")
    scores = reranker_model(**inputs).logits.squeeze().tolist()
    # Sort chunks by score in descending order and return top 3
    return [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)][:3]


def diagnose_with_gemini(query: str, context: list) -> str:
    context_text = "\n".join(context)
    prompt = f"""
    You are an expert dermatologist AI assistant. Your task is to diagnose a skin condition based on user symptoms and relevant literature.

    **Patient Symptoms:** {query}
    **Relevant Dermatology Texts:**
    {context_text}

    **Diagnosis:**
    - List possible conditions with confidence scores (e.g., 80% Eczema, 20% Psoriasis).
    - Suggest follow-up questions if necessary.
    - Provide recommended treatments.
    """
    response = genai.generate_text(model="gemini-pro", prompt=prompt)
    return response["choices"][0]["text"]


@app.post("/diagnose/")
def diagnose(symptoms: str):
    retrieved_chunks = search_pinecone(symptoms)
    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No relevant data found in Pinecone index.")
    top_chunks = rerank_chunks(symptoms, retrieved_chunks)
    diagnosis = diagnose_with_gemini(symptoms, top_chunks)
    return {"diagnosis": diagnosis}


# Start the FastAPI application with Uvicorn when running this script directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
