import os
import uuid
import re
import logging
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import fitz  # PyMuPDF for extracting text from PDFs
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Ensure API keys are set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise EnvironmentError("API keys for Pinecone and/or Gemini are not set in the environment variables.")

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dermatology-chunks"

try:
    available_indexes = pc.list_indexes().names()
    logger.info("Available Pinecone indexes: %s", available_indexes)
except Exception as e:
    logger.error("Error listing indexes: %s", e)
    raise HTTPException(status_code=500, detail=f"Error listing indexes: {e}")

if index_name not in available_indexes:
    try:
        pc.create_index(
            name=index_name,
            dimension=768,  # Gemini embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Use a supported region for your plan
            )
        )
        logger.info("Index created: %s", index_name)
    except Exception as e:
        logger.error("Error creating index: %s", e)
        raise HTTPException(status_code=500, detail=f"Error creating index: {e}")

try:
    index = pc.Index(index_name)
    logger.info("Pinecone index initialized: %s", index_name)
except Exception as e:
    logger.error("Error initializing index: %s", e)
    raise HTTPException(status_code=500, detail=f"Error initializing index: {e}")

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error("Error configuring Gemini API: %s", e)
    raise HTTPException(status_code=500, detail=f"Error configuring Gemini API: {e}")

# Load the reranker model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
    logger.info("Reranker model loaded successfully.")
except Exception as e:
    logger.error("Error loading reranker model: %s", e)
    raise HTTPException(status_code=500, detail=f"Error loading reranker model: {e}")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        # Load the PDF from bytes using fitz (PyMuPDF)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        logger.info("Extracted text length: %d", len(text))
        return text
    except Exception as e:
        logger.error("Error processing PDF: %s", e)
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {e}")


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list:
    """
    Chunk text into smaller pieces to avoid exceeding API limits.
    Reduced chunk size to handle Gemini's size constraints.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Skip extremely long sentences by breaking them further if needed
        if len(sentence) > 4000:  # Arbitrary cutoff for very long sentences
            words = sentence.split()
            for i in range(0, len(words), 50):  # Break into groups of 50 words
                sub_sentence = " ".join(words[i:i+50])
                if current_length + len(sub_sentence.split()) > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sub_sentence)
                current_length += len(sub_sentence.split())
            continue
            
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Removed overlap to avoid large chunks
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Additional check to ensure no chunk is too large for the API
    max_bytes = 35000  # Safe limit below the 36000 byte API limit
    final_chunks = []
    for chunk in chunks:
        if len(chunk.encode('utf-8')) > max_bytes:
            # If still too large, break it down further by characters
            logger.warning(f"Chunk too large ({len(chunk.encode('utf-8'))} bytes), breaking down further")
            text_pieces = []
            current_piece = ""
            for sentence in re.split(r'(?<=[.!?]) ', chunk):
                if len((current_piece + sentence).encode('utf-8')) < max_bytes:
                    current_piece += sentence + " "
                else:
                    if current_piece:
                        text_pieces.append(current_piece.strip())
                    current_piece = sentence + " "
            if current_piece:
                text_pieces.append(current_piece.strip())
            final_chunks.extend(text_pieces)
        else:
            final_chunks.append(chunk)

    logger.info("Generated %d text chunks", len(final_chunks))
    return final_chunks


def get_embedding(text: str, max_retries=3) -> list:
    """Get embedding with retry logic for API failures"""
    for attempt in range(max_retries):
        try:
            # Check text size before sending to API
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > 30000:  # Safe limit
                logger.warning(f"Text too large ({len(text_bytes)} bytes), truncating...")
                # Truncate text to safe size by finding the last sentence break
                safe_text = text.encode('utf-8')[:30000].decode('utf-8', errors='ignore')
                last_period = safe_text.rfind('.')
                if last_period > 0:
                    text = safe_text[:last_period+1]
                else:
                    text = safe_text
            
            embedding_model = "models/embedding-001"
            response = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response["embedding"]
            return embedding
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)  # Add a delay before retrying
            else:
                logger.error(f"Error generating embedding after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail=f"Error generating embedding: {e}")


def store_in_pinecone(text_chunks: list):
    try:
        vectors = []
        failed_chunks = 0
        
        for i, chunk in enumerate(text_chunks):
            try:
                embedding = get_embedding(chunk)
                vectors.append((f"chunk-{i}-{uuid.uuid4()}", embedding, {"text": chunk}))
                if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Processed {i + 1}/{len(text_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i}: {e}")
                failed_chunks += 1
                # Continue with other chunks instead of failing completely
                continue
        
        if not vectors:
            raise HTTPException(status_code=500, detail="Failed to generate any valid embeddings")
            
        # Upsert in smaller batches to avoid timeouts
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            logger.info(f"Upserting batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} with {len(batch)} vectors")
            
            # Add retry logic for Pinecone upserts
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    index.upsert(vectors=batch)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Upsert attempt {attempt+1} failed: {e}. Retrying...")
                        time.sleep(2)  # Add a delay before retrying
                    else:
                        logger.error(f"Failed to upsert batch after {max_retries} attempts")
                        raise
        
        logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")
        if failed_chunks > 0:
            logger.warning(f"{failed_chunks} chunks could not be embedded and were skipped")
            
    except Exception as e:
        logger.error("Error storing vectors in Pinecone: %s", e)
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
        logger.error("Error reading PDF: %s", e)
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

    # Use smaller chunks to avoid hitting API limits
    text_chunks = chunk_text(text)
    store_in_pinecone(text_chunks)
    return {"message": "PDF ingested successfully", "num_chunks": len(text_chunks)}


def search_pinecone(query_text: str, top_k: int = 5) -> list:
    try:
        query_embedding = get_embedding(query_text)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = results["matches"]
        if not matches:
            logger.warning("No matches found in Pinecone.")
            return []
        return [match["metadata"]["text"] for match in matches]
    except Exception as e:
        logger.error("Error searching Pinecone: %s", e)
        raise HTTPException(status_code=500, detail=f"Error searching Pinecone: {e}")


def rerank_chunks(query: str, chunks: list) -> list:
    if not chunks:
        logger.warning("No chunks to rerank.")
        return []
    try:
        # Prepare input pairs
        pairs = [[query, chunk] for chunk in chunks]
        
        # Tokenize and get scores
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            scores = reranker_model(**inputs).logits.squeeze().tolist()
        
        # Ensure scores is a list even if only one chunk
        if not isinstance(scores, list):
            scores = [scores]
        
        # Sort chunks by score in descending order and return top 3
        sorted_results = sorted(zip(scores, chunks), reverse=True)
        return [chunk for _, chunk in sorted_results][:3]
    except Exception as e:
        logger.error("Error reranking chunks: %s", e)
        # Return original chunks if reranking fails
        return chunks[:3]


# Define a Pydantic model for request validation
class DiagnosisRequest(BaseModel):
    symptoms: str


def diagnose_with_gemini(query: str, context: list) -> str:
    # Limit context to prevent exceeding Gemini's input limits
    max_context_length = 10000
    context_text = ""
    for chunk in context:
        if len(context_text) + len(chunk) < max_context_length:
            context_text += chunk + "\n\n"
        else:
            break
        
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
    try:
        # Create Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Extract text from response
        diagnosis_text = response.text
        return diagnosis_text
    except Exception as e:
        logger.error("Error generating diagnosis: %s", e)
        raise HTTPException(status_code=500, detail=f"Error generating diagnosis: {e}")


@app.post("/diagnose/")
async def diagnose(request: DiagnosisRequest):
    symptoms = request.symptoms
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

def list_available_models():
    try:
        models = genai.list_models()
        logger.info("Available models:")
        for model in models:
            logger.info(f"- {model.name}")
        return [model.name for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []

# Add this endpoint to your FastAPI app:
@app.get("/list_models/")
async def list_models():
    models = list_available_models()
    return {"available_models": models}