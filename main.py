import os
import pdfplumber
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from langchain_together import ChatTogether

app = FastAPI()

# Remove default folder; user will now supply folder path via the endpoint

# Load SentenceTransformer Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index and CV store will be initialized when the folder is loaded
embedding_dim = 384  # Embedding size of MiniLM-L6-v2
faiss_index = None
cv_store = {}

together_api_key = "3af858d7c0f222a22514e1cf0ac730cdb8848a17d4b667cf06f76bbfc9ca371b"

# Initialize the Together AI Chat Model
chat_model = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    together_api_key=together_api_key
)

# Pydantic models for incoming requests
class SearchQuery(BaseModel):
    query: str

class FolderPath(BaseModel):
    folder_path: str

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_folder(folder_path):
    """Loads all PDFs from the given folder and returns a FAISS index and CV store."""
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

@app.post("/load_folder/")
async def load_folder(folder: FolderPath):
    """Loads CVs from a provided folder path."""
    global faiss_index, cv_store
    folder_path = folder.folder_path
    if not os.path.isdir(folder_path):
        return {"error": "Provided folder path is not valid."}
    faiss_index, cv_store = process_folder(folder_path)
    return {"message": f"Loaded {len(cv_store)} CVs from {folder_path}"}

def refine_query(query):
    """Use Together AI to improve search queries."""
    response = chat_model.invoke(f"Rewrite this job description into a structured search query for candidate selection: {query}")
    if hasattr(response, 'content'):
        return response.content
    return response

class SearchRequest(BaseModel):
    folder_path: str
    query: str

@app.post("/search/")
async def search_cvs(request: SearchRequest):
    """Loads CVs from the provided folder and performs a search immediately."""
    global faiss_index, cv_store
    folder_path = request.folder_path

    # Validate folder path
    if not os.path.isdir(folder_path):
        return {"error": "Provided folder path is not valid."}

    # Load and process folder
    faiss_index, cv_store = process_folder(folder_path)

    # Ensure that PDFs were found in the provided folder
    if not cv_store:
        return {"error": "No PDFs found in the provided folder."}

    # ✨ Step 1: Improve the search query using Together AI
    refined_query = refine_query(request.query)

    # ✅ Extract content correctly from AIMessage
    if isinstance(refined_query, list) and len(refined_query) > 0:
        refined_query = refined_query[0].content  # Extract from list
    elif hasattr(refined_query, 'content'):
        refined_query = refined_query.content  # Extract from AIMessage

    print(f"Original Query: {request.query}")
    print(f"Refined Query: {refined_query}")

    # 🧠 Step 2: Generate an embedding for the improved query
    query_embedding = embedding_model.encode(refined_query, convert_to_numpy=True)

    # 🔍 Step 3: Search FAISS for top matches
    D, I = faiss_index.search(np.array([query_embedding]), k=5)

    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(cv_store):
            file_id = list(cv_store.keys())[idx]
            results.append({
                "filename": cv_store[file_id]["filename"],
                "score": float(D[0][i]),
                "text_snippet": cv_store[file_id]["text"]
            })

    return {
        "original_query": request.query,
        "refined_query": refined_query,
        "results": results
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a PDF file, extracts text, and stores embeddings into the loaded folder context."""
    global faiss_index, cv_store
    if faiss_index is None:
        return {"error": "No folder loaded. Please load a folder using the /load_folder/ endpoint."}
    file_id = str(uuid.uuid4())
    # Save file temporarily
    temp_path = os.path.join(os.getcwd(), file.filename)
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    extracted_text = extract_text_from_pdf(temp_path)
    embedding = embedding_model.encode(extracted_text, convert_to_numpy=True)
    faiss_index.add(np.array([embedding]))
    cv_store[file_id] = {"filename": file.filename, "text": extracted_text[:300]}
    os.remove(temp_path)  # Clean up the temporary file
    return {"file_id": file_id, "message": "File uploaded and processed successfully."}
