from fastapi import FastAPI, HTTPException
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import fitz  # PyMuPDF for extracting text from PDFs
import os
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Connect to Milvus (Make sure Milvus is running)
connections.connect("default", host="localhost", port="19530")

# Define Milvus collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048)
]
schema = CollectionSchema(fields, description="Dermatology Literature Embeddings")

collection_name = "dermatology_chunks"
if collection_name not in Collection.list_collections():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)
collection.load()

# Initialize Gemini API key
GEMINI_API_KEY = os.getenv("AIzaSyDyInCCobdHSKGkHE7hYouspUrQpmAfxdo")
genai.configure(api_key=GEMINI_API_KEY)

# Load reranker model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_embedding(text):
    response = genai.embed_content(model="text-embedding-gecko", content=text)
    return response["embedding"]

def store_in_milvus(text_chunks):
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    collection.insert([[i for i in range(len(embeddings))], embeddings, text_chunks])
    collection.load()

@app.post("/ingest_pdf/")
def ingest_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(text)
    store_in_milvus(text_chunks)
    return {"message": "PDF ingested successfully", "num_chunks": len(text_chunks)}

def search_milvus(query_text, top_k=5):
    query_embedding = get_embedding(query_text)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding], anns_field="embedding", param=search_params, limit=top_k
    )
    return [hit.entity.get("text") for hit in results[0]]

def rerank_chunks(query, chunks):
    inputs = tokenizer([query] + chunks, padding=True, truncation=True, return_tensors="pt")
    scores = reranker_model(**inputs).logits.squeeze().tolist()
    return [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)][:3]

def diagnose_with_gemini(query, context):
    prompt = f"Patient symptoms: {query}\n\nRelevant dermatology texts:\n{context}\n\nWhat is the most likely diagnosis?"
    response = genai.generate_text(model="gemini-pro", prompt=prompt)
    return response["choices"][0]["text"]

@app.post("/diagnose/")
def diagnose(symptoms: str):
    retrieved_chunks = search_milvus(symptoms)
    top_chunks = rerank_chunks(symptoms, retrieved_chunks)
    diagnosis = diagnose_with_gemini(symptoms, top_chunks)
    return {"diagnosis": diagnosis}
