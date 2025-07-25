from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from dotenv import load_dotenv
load_dotenv()


class VectorStore:
    """Simple wrapper around Chroma's `PersistentClient` for HR document embeddings."""

    def __init__(
        self,
        collection_name: str = "hr_docs",
        persist_directory: str | os.PathLike[str] = "data/chroma_db",
    ) -> None:
        # Ensure the persistence directory exists relative to project root
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Create a persistent client
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection_name = collection_name

        # Create or get existing collection
        try:
            self._collection = self._client.get_collection(collection_name)
        except chromadb.errors.NotFoundError:
            self._collection = self._client.create_collection(collection_name)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents + embeddings to the collection."""
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids],
        )

    def query(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Dict:
        """Query similar documents, optionally with metadata filters."""
        query_params = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
        }
        if filters:
            query_params["where"] = filters
        return self._collection.query(**query_params)

    # Convenience factory --------------------------------------------------
    @classmethod
    def default(cls) -> "VectorStore":
        """Get a default instance with default params."""
        return cls() 