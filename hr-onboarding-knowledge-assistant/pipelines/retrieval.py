from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from services.vector_store import VectorStore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _placeholder_embedding(text: str, dims: int = 768) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeat_times = (dims // len(digest)) + 1
    long_bytes = (digest * repeat_times)[:dims]
    return [b / 255.0 for b in long_bytes]


def _categorize_query(question: str) -> str:
    lowered = question.lower()
    if any(k in lowered for k in ("vacation", "leave", "pto", "holiday")):
        return "leave"
    if any(k in lowered for k in ("benefit", "insurance", "health", "dental")):
        return "benefits"
    if any(k in lowered for k in ("conduct", "code of conduct", "policy")):
        return "conduct"
    return "general"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def answer_query(query: str, top_k: int = 3) -> Dict:
    """Return answer dict with response, citations, category."""
    vector_store = VectorStore.default()
    query_emb = _placeholder_embedding(query)
    # Query the vector store – if the collection is empty, Chroma raises `ValueError`.
    try:
        results = vector_store.query([query_emb], top_k=top_k)
    except Exception:
        # Gracefully handle empty or missing collection without crashing the API.
        return {
            "response": "Sorry, I couldn't find relevant information in the uploaded HR documents.",
            "citations": [],
            "category": _categorize_query(query),
        }

    # Unpack results (chromadb returns dict)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return {
            "response": "Sorry, I couldn't find relevant information in the uploaded HR documents.",
            "citations": [],
            "category": _categorize_query(query),
        }

    # For placeholder, just return first chunk
    answer_chunk = docs[0]
    citation = metadatas[0].get("source_file") if metadatas else None

    return {
        "response": answer_chunk.strip(),
        "citations": [citation] if citation else [],
        "category": _categorize_query(query),
    } 