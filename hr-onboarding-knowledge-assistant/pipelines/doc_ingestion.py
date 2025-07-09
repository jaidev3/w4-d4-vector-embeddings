from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

# External deps for parsing
from pypdf import PdfReader  # type: ignore
from docx import Document as DocxDocument  # type: ignore

from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers for text extraction
# -----------------------------------------------------------------------------

def _extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_text_from_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def _extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


action_map = {
    ".pdf": _extract_text_from_pdf,
    ".docx": _extract_text_from_docx,
    ".txt": _extract_text_from_txt,
}


# -----------------------------------------------------------------------------
# Embedding (placeholder)
# -----------------------------------------------------------------------------

def _placeholder_embedding(text: str, dims: int = 768) -> List[float]:
    """Generate deterministic pseudo-embedding by hashing the text.

    This is **NOT** suitable for production – replace with real embeddings later.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat / truncate digest to match dims
    repeat_times = (dims // len(digest)) + 1
    long_bytes = (digest * repeat_times)[:dims]
    # Map bytes to floats in range [0,1]
    return [b / 255.0 for b in long_bytes]


# -----------------------------------------------------------------------------
# Main ingestion function
# -----------------------------------------------------------------------------

def ingest_files(file_paths: List[str], vector_store: VectorStore | None = None) -> Dict[str, int]:
    """Ingest provided files and add their embeddings to the vector store.

    Returns a mapping of file path -> number of chunks indexed.
    """
    if vector_store is None:
        vector_store = VectorStore.default()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )

    stats: Dict[str, int] = {}

    for file_path in file_paths:
        path = Path(file_path)
        ext = path.suffix.lower()
        extractor = action_map.get(ext)
        if extractor is None:
            logger.warning("Skipping unsupported file type: %s", path)
            continue

        logger.info("Extracting text from %s", path)
        try:
            raw_text = extractor(path)
        except Exception as exc:
            logger.exception("Failed to extract text from %s: %s", path, exc)
            continue

        chunks = splitter.split_text(raw_text)
        if not chunks:
            logger.warning("No text extracted from %s", path)
            continue

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{path.name}-{idx}"
            ids.append(chunk_id)
            embeddings.append(_placeholder_embedding(chunk))
            metadatas.append({
                "source_file": path.name,
                "chunk_index": idx,
                "ext": ext,
            })

        vector_store.add_embeddings(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        stats[file_path] = len(chunks)
        logger.info("Indexed %s chunks from %s", len(chunks), path)

    return stats 