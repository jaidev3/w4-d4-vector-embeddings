# HR Onboarding Knowledge Assistant

> Replace time-consuming HR induction calls with an AI assistant that lets new employees instantly query company policies, benefits, leave rules, and more – all sourced from your own HR documents.

---

## 📚 What this project does

1. **Document Upload** – HR can upload PDFs, DOCX, or TXT files (`/upload` endpoint or *Upload* tab in the Streamlit UI).
2. **Ingestion & Indexing** – Text is extracted, chunked, and (for now) converted into *placeholder* embeddings. Chunks + embeddings are stored persistently in a local Chroma vector database.
3. **Conversational Querying** – Employees ask questions via chat. Their query is embedded, the most relevant chunks are retrieved, and a response is generated with policy citations.
4. **Admin Dashboard** – A lightweight Streamlit dashboard lets HR delete previously uploaded files.

> ⚠️  The current version uses deterministic **placeholder embeddings** (fast but low-quality). Swap in a real embedding model (e.g. OpenAI, HuggingFace, instructor-XL, etc.) to improve answer quality – no other code changes required.

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
pip install streamlit fastapi uvicorn langchain chromadb tiktoken python-docx pypdf python-multipart
```

(Feel free to capture these into a `requirements.txt` for CI/CD.)

### 4. Start the FastAPI backend

```bash
uvicorn backend.app:app --reload --port 8000
```
The API will be accessible at `http://localhost:8000` (interactive docs at `/docs`).

### 5. Launch the Streamlit employee UI

Open **a second terminal** (keep the backend running) and run:

```bash
streamlit run streamlit_app.py
```
This opens a browser tab with two tabs:
* **📄 Upload HR Documents** – choose files & click *Upload & Ingest*
* **💬 Chat** – ask questions; answers include citations to source files

### 6. (Optional) Launch the admin dashboard

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

## 🧩  Swapping in real embeddings

1. Replace `_placeholder_embedding` in `pipelines/doc_ingestion.py` and `pipelines/retrieval.py` with a call to your embedding provider (OpenAI, Cohere, HuggingFace, etc.).
2. Keep the embedding dimensionality consistent between ingestion & retrieval.
3. Re-run the ingestion step (delete `data/chroma_db/` if you want a clean index).

---

## 🚦  API reference (FastAPI)

| Method | Path | Body / Query | Description |
|--------|------|-------------|-------------|
| `POST` | `/upload` | `multipart/form-data` with one or more `files` | Upload and ingest documents; returns stats |
| `POST` | `/chat` | `{ "query": "…" }` | Returns `{ response, citations }` |

Swagger docs: http://localhost:8000/docs

---

## 📝  TODOs & future improvements

- Swap placeholder embeddings for production-grade model
- Summarise multiple matching chunks for more coherent answers
- User authentication for admin dashboard
- Dockerfile / k8s manifests for production deployment
- CI tests

PRs welcome – happy onboarding! 🎉 