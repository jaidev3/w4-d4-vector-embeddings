import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="HR Onboarding Knowledge Assistant API")

# Enable CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for saving uploaded HR documents
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Pydantic models
# ----------------------------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    citations: List[str] = []

# ----------------------------
# Routes
# ----------------------------

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):  # noqa: B008
    """Upload HR documents (PDF, DOCX, TXT)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved_paths: List[str] = []
    for uploaded_file in files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
        contents = await uploaded_file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        saved_paths.append(file_path)

    # Trigger ingestion pipeline
    try:
        from pipelines.doc_ingestion import ingest_files  # Local import to avoid slow startup

        stats = ingest_files(saved_paths)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {"message": "Files uploaded and ingested", "files": saved_paths, "stats": stats}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle employee query and return answer with citations."""
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        from pipelines.retrieval import answer_query  # Local import to avoid slow startup

        result = answer_query(user_query)
        return ChatResponse(response=result["response"], citations=result["citations"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc


# ----------------------------
# Uvicorn entrypoint (optional)
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True) 