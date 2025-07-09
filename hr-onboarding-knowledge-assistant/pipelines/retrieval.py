from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from openai import OpenAI
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# OpenAI Embedding
# -----------------------------------------------------------------------------

def _get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embeddings using OpenAI's embedding API.
    
    Args:
        text: The text to embed
        model: OpenAI embedding model to use (default: text-embedding-3-small)
    
    Returns:
        List of floats representing the embedding vector
    
    Raises:
        Exception: If OpenAI API call fails
    """
    client = OpenAI(api_key="add-your-key-here")
    
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as exc:
        logger.error("Failed to generate OpenAI embedding: %s", exc)
        raise exc


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
    
    try:
        query_emb = _get_openai_embedding(query)
    except Exception as exc:
        logger.error("Failed to generate query embedding: %s", exc)
        return {
            "response": "Sorry, I encountered an error processing your query. Please try again.",
            "citations": [],
            "category": _categorize_query(query),
        }
    
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

    # For now, just return first chunk
    answer_chunk = docs[0]
    citation = metadatas[0].get("source_file") if metadatas else None

    return {
        "response": answer_chunk.strip(),
        "citations": [citation] if citation else [],
        "category": _categorize_query(query),
    } 