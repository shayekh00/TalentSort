import os
import pdfplumber
import uuid
import numpy as np
import spacy
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from langchain_together import ChatTogether
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

app = FastAPI()

# Load Spacy NER model
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index and CV Store
embedding_dim = 384  # MiniLM embedding size
faiss_index = None
cv_store = {}

# API Keys
serpapi_key = os.getenv("SERPAPI_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

if not together_api_key:
    raise ValueError("❌ ERROR: TOGETHER_API_KEY not found in .env file!")

if not serpapi_key:
    raise ValueError("❌ ERROR: SERPAPI_KEY not found in .env file!")

# Initialize Together AI Chat Model
chat_model = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    together_api_key=together_api_key
)

class SearchRequest(BaseModel):
    folder_path: str
    query: str

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_folder(folder_path):
    """Loads PDFs from a folder and stores their embeddings."""
    index = faiss.IndexFlatL2(embedding_dim)
    store = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            file_id = str(uuid.uuid4())
            extracted_text = extract_text_from_pdf(file_path)
            embedding = embedding_model.encode(extracted_text, convert_to_numpy=True)
            index.add(np.array([embedding]))
            store[file_id] = {"filename": filename, "text": extracted_text[:300]}
    return index, store

def extract_company_names(text):
    """Extract company names using NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def get_company_reputation(company_name):
    """Fetch company reputation using SerpAPI."""
    search = GoogleSearch({
        "q": f"{company_name} company review OR Glassdoor rating OR funding OR reputation",
        "api_key": serpapi_key
    })
    results = search.get_dict()
    return results["organic_results"][0]["snippet"] if "organic_results" in results else "No data available"

def compute_final_score(semantic_score, experience_score, company_score):
    """Compute the final ranking score."""
    return (0.6 * semantic_score) + (0.2 * experience_score) + (0.2 * company_score)

@app.post("/search/")
async def search_cvs(request: SearchRequest):
    """Loads CVs from a folder and performs a search immediately."""
    global faiss_index, cv_store
    folder_path = request.folder_path

    if not os.path.isdir(folder_path):
        return {"error": "Invalid folder path."}

    faiss_index, cv_store = process_folder(folder_path)
    if not cv_store:
        return {"error": "No PDFs found."}

    refined_query = chat_model.invoke(f"Rewrite this job description: {request.query}")
    refined_query = refined_query.content if hasattr(refined_query, 'content') else refined_query

    query_embedding = embedding_model.encode(refined_query, convert_to_numpy=True)
    D, I = faiss_index.search(np.array([query_embedding]), k=5)

    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(cv_store):
            file_id = list(cv_store.keys())[idx]
            resume_text = cv_store[file_id]["text"]
            filename = cv_store[file_id]["filename"]
            
            company_names = extract_company_names(resume_text)
            company_score = sum(0.2 for company in company_names if "4." in get_company_reputation(company))
            
            experience_score = 0
            years_experience = [int(num) for num in resume_text.split() if num.isdigit() and int(num) < 50]
            if years_experience:
                experience_score = min(max(years_experience) / 20, 1)

            final_score = compute_final_score(D[0][i], experience_score, company_score)

            results.append({
                "filename": filename,
                "score": float(final_score),  # Convert NumPy float32 to Python float
                "semantic_score": float(D[0][i]),  # Ensure conversion
                "experience_score": float(experience_score),  # Ensure conversion
                "company_score": float(company_score),  # Ensure conversion
                "text_snippet": resume_text[:300]
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "original_query": request.query,
        "refined_query": refined_query,
        "results": results
    }
