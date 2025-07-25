# HR Onboarding Knowledge Assistant

> Replace time-consuming HR induction calls with an AI assistant that lets new employees instantly query company policies, benefits, leave rules, and more – all sourced from your own HR documents.

---

## 📚 What this project does

1. **Document Upload** – HR can upload PDFs, DOCX, or TXT files (`/upload` endpoint or *Upload* tab in the Streamlit UI).
2. **Ingestion & Indexing** – Text is extracted, chunked, and converted into high-quality OpenAI embeddings. Chunks + embeddings are stored persistently in a local Chroma vector database.
3. **Conversational Querying** – Employees ask questions via chat. Their query is embedded, the most relevant chunks are retrieved, and a response is generated with policy citations.
4. **Admin Dashboard** – A lightweight Streamlit dashboard lets HR delete previously uploaded files.

> ✅  This version uses **OpenAI's text-embedding-3-small model** for high-quality semantic embeddings. Make sure to set your `OPENAI_API_KEY` environment variable.

---

## 🛠️  Quick-start

### 1. Clone & enter the project

```bash
# already cloned if you are reading this
cd hr-onboarding-knowledge-assistant
```

### 2. Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 4. Set up OpenAI API key

Create a `.env` file in the project root and add your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or export it as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Test your setup:**
```bash
python demo_embeddings.py
```
This will verify your OpenAI API key is working and show you how the embeddings work.

### 5. Start the FastAPI backend

```bash
uvicorn backend.app:app --reload --port 8000
```
The API will be accessible at `http://localhost:8000` (interactive docs at `/docs`).

### 6. Launch the Streamlit employee UI

Open **a second terminal** (keep the backend running) and run:

```bash
streamlit run streamlit_app.py
```
This opens a browser tab with two tabs:
* **📄 Upload HR Documents** – choose files & click *Upload & Ingest*
* **💬 Chat** – ask questions; answers include citations to source files

### 7. (Optional) Launch the admin dashboard

```bash
streamlit run admin_dashboard.py
```
Here HR staff can see uploaded files and delete them if needed.

---

## 🔍  How it works under the hood

| Step | Module | Description |
|------|--------|-------------|
| 1 | `backend/app.py` | FastAPI server exposing `/upload` and `/chat` endpoints |
| 2 | `pipelines/doc_ingestion.py` | Extracts text, splits into overlap-aware chunks, generates embeddings, stores in Chroma |
| 3 | `services/vector_store.py` | Thin wrapper around persistent Chroma client |
| 4 | `pipelines/retrieval.py` | Given a user query, embeds it, retrieves similar chunks, picks best answer, returns citations & category |
| 5 | `streamlit_app.py` | Front-end for both uploading and chatting |
| 6 | `admin_dashboard.py` | Simple Streamlit dashboard for file management |

![architecture](https://gist.githubusercontent.com/your-name/placeholder-diagram/raw/architecture.png)

---

## 🧩  OpenAI Embeddings

This project now uses **OpenAI's text-embedding-3-small model** for high-quality semantic embeddings. The embeddings are generated in real-time during both document ingestion and query processing.

**Key features:**
- Uses OpenAI's latest embedding model (text-embedding-3-small)
- 1536-dimensional embeddings for better semantic understanding
- Automatic error handling and logging
- Consistent embedding generation across ingestion and retrieval

**To use a different embedding model:**
1. Modify the `model` parameter in `_get_openai_embedding()` functions
2. Update both `pipelines/doc_ingestion.py` and `pipelines/retrieval.py`
3. Re-run the ingestion step (delete `data/chroma_db/` if you want a clean index)

---

## 🚦  API reference (FastAPI)

| Method | Path | Body / Query | Description |
|--------|------|-------------|-------------|
| `POST` | `/upload` | `multipart/form-data` with one or more `files` | Upload and ingest documents; returns stats |
| `POST` | `/chat` | `{ "query": "…" }` | Returns `{ response, citations }` |

Swagger docs: http://localhost:8000/docs

---

## 📝  TODOs & future improvements

- ✅ ~~Swap placeholder embeddings for production-grade model~~ (Now using OpenAI embeddings)
- Summarise multiple matching chunks for more coherent answers
- User authentication for admin dashboard
- Dockerfile / k8s manifests for production deployment
- CI tests
- Add support for .env file loading (python-dotenv)
- Implement batch processing for large document uploads
- Add embedding model configuration options

PRs welcome – happy onboarding! 🎉 

## API Keys Setup

Create a `.env` file in the `hr-onboarding-knowledge-assistant` directory with the following content:

```
# Pinecone API Key
PINECONE_API_KEY=your-pinecone-api-key

# Gemini API Key
GEMINI_API_KEY=your-gemini-api-key
```

Replace `your-pinecone-api-key` and `your-gemini-api-key` with your actual API keys. 