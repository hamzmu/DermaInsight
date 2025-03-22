from fastapi import FastAPI, HTTPException
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

# Initialize FastAPI
app = FastAPI()

# Load API keys securely from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dermatology-chunks"

# Check if index exists and create it if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Adjust to the embedding dimension of your model
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',  # Adjust the cloud provider and region
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load reranker model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=512, overlap=128):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-(overlap // 2):]  # Keep some overlap
            current_length = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embedding(text):
    response = genai.embed_content(model="text-embedding-gecko", content=text)
    return response["embedding"]

def store_in_pinecone(text_chunks):
    vectors = [(str(uuid.uuid4()), get_embedding(chunk), {"text": chunk}) for chunk in text_chunks]
    index.upsert(vectors)

@app.post("/ingest_pdf/")
def ingest_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(text)
    store_in_pinecone(text_chunks)
    return {"message": "PDF ingested successfully", "num_chunks": len(text_chunks)}

def search_pinecone(query_text, top_k=5):
    query_embedding = get_embedding(query_text)
    results = index.query(queries=[query_embedding], top_k=top_k, include_metadata=True)
    return [match.metadata["text"] for match in results["matches"]]

def rerank_chunks(query, chunks):
    inputs = tokenizer([query] + chunks, padding=True, truncation=True, return_tensors="pt")
    scores = reranker_model(**inputs).logits.squeeze().tolist()
    return [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)][:3]

def diagnose_with_gemini(query, context):
    prompt = f"""
    You are an expert dermatologist AI assistant. Your task is to diagnose a skin condition based on user symptoms and relevant literature.
    
    **Patient Symptoms:** {query}
    **Relevant Dermatology Texts:**
    {context}
    
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
    top_chunks = rerank_chunks(symptoms, retrieved_chunks)
    diagnosis = diagnose_with_gemini(symptoms, top_chunks)
    return {"diagnosis": diagnosis}

