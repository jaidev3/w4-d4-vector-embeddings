import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "]
        )
    
    def extract_text_from_pdf(self, path: Path) -> str:
        """Extract text content from a PDF file"""
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into smaller chunks for embedding"""
        return self.text_splitter.split_text(text)

class EmbeddingService:
    """Handles OpenAI embedding generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embedding for given text using OpenAI API"""
        response = self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

class VectorStore:
    """Manages ChromaDB operations for document storage and retrieval"""
    
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "research_papers"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy initialization of ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client
    
    @property
    def collection(self):
        """Get or create the ChromaDB collection"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                self._collection = self.client.create_collection(self.collection_name)
        return self._collection
    
    def add_documents(self, ids: List[str], embeddings: List[List[float]], 
                     documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents with their embeddings to the vector store"""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Retrieve all documents grouped by source file"""
        try:
            results = self.collection.get()
            
            # Group by source file
            documents_by_file = {}
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                source_file = metadata.get('source_file', 'Unknown')
                
                if source_file not in documents_by_file:
                    documents_by_file[source_file] = {
                        'chunks': [],
                        'total_chunks': 0
                    }
                
                documents_by_file[source_file]['chunks'].append({
                    'id': doc_id,
                    'content': results['documents'][i] if results['documents'] else '',
                    'metadata': metadata
                })
                documents_by_file[source_file]['total_chunks'] += 1
                
            return documents_by_file
        except Exception as e:
            raise Exception(f"Error retrieving documents: {e}")
    
    def delete_document_by_source(self, source_file: str) -> int:
        """Delete all chunks for a specific source file"""
        try:
            results = self.collection.get()
            
            # Find all IDs for this source file
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata and metadata.get('source_file') == source_file:
                    ids_to_delete.append(results['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            else:
                return 0
                
        except Exception as e:
            raise Exception(f"Error deleting document: {e}")
    
    def delete_all_documents(self) -> int:
        """Delete all documents from the collection"""
        try:
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            else:
                return 0
        except Exception as e:
            raise Exception(f"Error deleting all documents: {e}")

class ResearchAssistant:
    """Main class that orchestrates the research assistant functionality"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
    
    def process_pdf_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process multiple PDF files and index them in the vector store"""
        results = {
            'total_files': len(file_paths),
            'processed_files': [],
            'failed_files': [],
            'total_chunks': 0
        }
        
        for file_path in file_paths:
            try:
                # Extract text from PDF
                raw_text = self.pdf_processor.extract_text_from_pdf(file_path)
                
                # Split into chunks
                chunks = self.pdf_processor.split_text_into_chunks(raw_text)
                
                # Generate embeddings and prepare data
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{file_path.name}-{idx}"
                    try:
                        embedding = self.embedding_service.get_embedding(chunk)
                        ids.append(chunk_id)
                        embeddings.append(embedding)
                        documents.append(chunk)
                        metadatas.append({
                            "source_file": file_path.name,
                            "chunk_index": idx,
                            "chunk_length": len(chunk)
                        })
                    except Exception as e:
                        print(f"Warning: Embedding failed for chunk {chunk_id}: {e}")
                        continue
                
                # Add to vector store
                if embeddings:
                    self.vector_store.add_documents(ids, embeddings, documents, metadatas)
                    results['processed_files'].append({
                        'name': file_path.name,
                        'chunks': len(embeddings)
                    })
                    results['total_chunks'] += len(embeddings)
                else:
                    results['failed_files'].append({
                        'name': file_path.name,
                        'error': 'No embeddings generated'
                    })
                    
            except Exception as e:
                results['failed_files'].append({
                    'name': file_path.name,
                    'error': str(e)
                })
        
        return results
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents from the vector store"""
        return self.vector_store.get_all_documents()
    
    def delete_document(self, source_file: str) -> int:
        """Delete a specific document by source file name"""
        return self.vector_store.delete_document_by_source(source_file)
    
    def delete_all_documents(self) -> int:
        """Delete all documents from the vector store"""
        return self.vector_store.delete_all_documents()
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents based on query (future enhancement)"""
        # This method can be implemented for semantic search functionality
        # For now, it's a placeholder for future development
        pass
