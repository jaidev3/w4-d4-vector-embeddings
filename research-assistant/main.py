import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
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

class WebSearchService:
    """Handles web search using Tavily through LangChain"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.search_tool = None
        if self.api_key:
            try:
                self.search_tool = TavilySearch(
                    max_results=5,
                    search_depth="advanced",
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Tavily search: {e}")
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web for information related to the query"""
        if not self.search_tool:
            return {
                "error": "Tavily API key not found or search tool not initialized. Please set TAVILY_API_KEY environment variable.",
                "results": []
            }
        
        try:
            results = self.search_tool.invoke({"query": query})
            
            # Format results for consistency with document search
            # TavilySearch returns a dict with 'results' key containing list of results
            search_results = results.get("results", [])
            formatted_results = []
            
            for result in search_results:
                formatted_results.append({
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0)
                })
            
            return {
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "error": f"Web search failed: {str(e)}",
                "results": []
            }

class LLMService:
    """Handles OpenAI chat completions for answer generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Generate an answer based on the query and retrieved context chunks"""
        try:
            # Prepare context from chunks
            context_text = ""
            sources = set()
            
            for i, chunk in enumerate(context_chunks):
                context_text += f"\n--- Document Chunk {i+1} ---\n"
                context_text += chunk.get('content', '')
                
                # Collect source information
                metadata = chunk.get('metadata', {})
                source_file = metadata.get('source_file', 'Unknown')
                sources.add(source_file)
            
            # Create the prompt
            system_prompt = """You are a helpful research assistant. Your task is to answer questions based on the provided document chunks. 

Instructions:
1. Use only the information provided in the document chunks to answer the question
2. If the information is not sufficient to answer the question, say so clearly
3. Provide specific quotes or references when possible
4. Structure your answer clearly and concisely
5. If multiple perspectives or details are available, present them comprehensively"""

            user_prompt = f"""Question: {query}

Context from documents:
{context_text}

Please provide a comprehensive answer based on the above context."""

            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more focused answers
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": list(sources),
                "context_chunks_used": len(context_chunks),
                "model_used": model
            }
            
        except Exception as e:
            return {
                "error": f"Error generating answer: {str(e)}",
                "answer": None,
                "sources": [],
                "context_chunks_used": 0
            }
    
    def generate_web_answer(self, query: str, web_results: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Generate an answer based on web search results"""
        try:
            # Prepare context from web results
            context_text = ""
            sources = set()
            
            for i, result in enumerate(web_results):
                context_text += f"\n--- Web Result {i+1} ---\n"
                context_text += f"Title: {result.get('title', 'No title')}\n"
                context_text += f"URL: {result.get('url', 'No URL')}\n"
                context_text += f"Content: {result.get('content', '')}\n"
                
                # Collect source URLs
                url = result.get('url', '')
                if url:
                    sources.add(url)
            
            # Create the prompt for web-based answers
            system_prompt = """You are a helpful research assistant. Your task is to answer questions based on web search results.

Instructions:
1. Use the information from the web search results to provide a comprehensive answer
2. Cite specific sources when possible by mentioning the website or URL
3. If the web results don't contain sufficient information, acknowledge this
4. Structure your answer clearly and provide context from multiple sources when available
5. Be objective and present information from the search results accurately"""

            user_prompt = f"""Question: {query}

Web search results:
{context_text}

Please provide a comprehensive answer based on the above web search results."""

            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": list(sources),
                "context_chunks_used": len(web_results),
                "model_used": model,
                "search_type": "web"
            }
            
        except Exception as e:
            return {
                "error": f"Error generating web-based answer: {str(e)}",
                "answer": None,
                "sources": [],
                "context_chunks_used": 0
            }

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

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store for the most similar documents to the query embedding."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            raise Exception(f"Error querying vector store: {e}")

class ResearchAssistant:
    """Main class that orchestrates the research assistant functionality"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.web_search_service = WebSearchService()
    
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
    
    def _should_fallback_to_web_search(self, answer: str, query: str) -> bool:
        """
        Determine if we should fallback to web search based on the LLM response.
        Returns True if the answer indicates the documents don't contain relevant information.
        """
        if not answer:
            return True
            
        # Check for common phrases that indicate the documents don't contain relevant information
        fallback_indicators = [
            "do not contain any information",
            "does not contain information",
            "cannot provide an answer",
            "cannot answer",
            "no information",
            "not mentioned",
            "not provided",
            "not available in the",
            "based on the available context",
            "the provided document chunks do not",
            "the documents do not contain",
            "i cannot provide",
            "there is no information",
            "no relevant information"
        ]
        
        answer_lower = answer.lower()
        
        # Check if any fallback indicators are present in the answer
        for indicator in fallback_indicators:
            if indicator in answer_lower:
                return True
        
        # Check if the answer is very short and doesn't seem to contain useful information
        if len(answer.strip()) < 50 and any(phrase in answer_lower for phrase in ["no", "not", "cannot", "unable"]):
            return True
            
        return False
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents based on query and generate an LLM answer."""
        # Validate input
        if not query or not query.strip():
            return {
                "error": "Query cannot be empty",
                "answer": None,
                "sources": [],
                "chunks": []
            }
        
        try:
            # Check if OpenAI API key is available
            if not self.embedding_service.api_key:
                return {
                    "error": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
                    "answer": None,
                    "sources": [],
                    "chunks": []
                }
            
            # Get embedding for the query
            try:
                query_embedding = self.embedding_service.get_embedding(query)
            except Exception as e:
                return {
                    "error": f"Failed to generate embedding: {str(e)}",
                    "answer": None,
                    "sources": [],
                    "chunks": []
                }
            
            if not query_embedding:
                return {
                    "error": "Failed to generate embedding for the query",
                    "answer": None,
                    "sources": [],
                    "chunks": []
                }
            
            # Query the vector store
            try:
                results = self.vector_store.query(query_embedding, n_results=n_results)
            except Exception as e:
                return {
                    "error": f"Vector store query failed: {str(e)}",
                    "answer": None,
                    "sources": [],
                    "chunks": []
                }
            
            if not results:
                return {
                    "answer": "No relevant documents found in the database.",
                    "sources": [],
                    "chunks": [],
                    "context_chunks_used": 0
                }
            
            # Format results for processing
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            ids = results.get("ids", [[]])[0] if results.get("ids") else []
            
            if not docs:
                # No documents found in the database, try web search as fallback
                try:
                    web_search_results = self.web_search_service.search_web(query)
                    
                    if web_search_results.get("error"):
                        return {
                            "answer": f"No relevant documents found in the database. Web search also failed: {web_search_results['error']}",
                            "sources": [],
                            "chunks": [],
                            "context_chunks_used": 0,
                            "search_type": "fallback_failed"
                        }
                    
                    web_results = web_search_results.get("results", [])
                    if not web_results:
                        return {
                            "answer": "No relevant documents found in the database and no web search results available.",
                            "sources": [],
                            "chunks": [],
                            "context_chunks_used": 0,
                            "search_type": "no_results"
                        }
                    
                    # Generate answer using web search results
                    llm_response = self.llm_service.generate_web_answer(query, web_results)
                    
                    # Format web results as chunks for consistency
                    web_chunks = [
                        {
                            "id": f"web_{i}",
                            "content": result["content"],
                            "metadata": {
                                "source_file": result["url"],
                                "title": result["title"],
                                "search_type": "web"
                            }
                        }
                        for i, result in enumerate(web_results)
                    ]
                    
                    return {
                        "answer": llm_response.get("answer"),
                        "sources": llm_response.get("sources", []),
                        "chunks": web_chunks,
                        "context_chunks_used": llm_response.get("context_chunks_used", 0),
                        "model_used": llm_response.get("model_used", ""),
                        "search_type": "web_fallback",
                        "error": llm_response.get("error")
                    }
                    
                except Exception as e:
                    return {
                        "answer": f"No relevant documents found in the database. Web search fallback failed: {str(e)}",
                        "sources": [],
                        "chunks": [],
                        "context_chunks_used": 0,
                        "search_type": "fallback_error"
                    }
            
            # Prepare chunks for LLM
            chunks = [
                {
                    "id": ids[i] if i < len(ids) else None,
                    "content": docs[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                for i in range(len(docs))
            ]
            
            # Generate answer using LLM
            try:
                llm_response = self.llm_service.generate_answer(query, chunks)
            except Exception as e:
                return {
                    "error": f"LLM service failed: {str(e)}",
                    "answer": None,
                    "sources": [],
                    "chunks": chunks
                }
            
            if not llm_response:
                return {
                    "error": "Failed to generate LLM response",
                    "answer": None,
                    "sources": [],
                    "chunks": chunks
                }
            
            # Check if the LLM indicates that the documents don't contain relevant information
            answer = llm_response.get("answer", "")
            if self._should_fallback_to_web_search(answer, query):
                # Document search didn't provide relevant information, try web search
                try:
                    web_search_results = self.web_search_service.search_web(query)
                    
                    if web_search_results.get("error"):
                        # Return the original document-based answer since web search failed
                        return {
                            "answer": llm_response.get("answer"),
                            "sources": llm_response.get("sources", []),
                            "chunks": chunks,
                            "context_chunks_used": llm_response.get("context_chunks_used", 0),
                            "model_used": llm_response.get("model_used", ""),
                            "search_type": "document_fallback_failed",
                            "error": f"Web search fallback failed: {web_search_results['error']}"
                        }
                    
                    web_results = web_search_results.get("results", [])
                    if not web_results:
                        # Return the original document-based answer since no web results
                        return {
                            "answer": llm_response.get("answer"),
                            "sources": llm_response.get("sources", []),
                            "chunks": chunks,
                            "context_chunks_used": llm_response.get("context_chunks_used", 0),
                            "model_used": llm_response.get("model_used", ""),
                            "search_type": "document_no_web_results",
                            "error": "No web search results available"
                        }
                    
                    # Generate answer using web search results
                    web_llm_response = self.llm_service.generate_web_answer(query, web_results)
                    
                    # Format web results as chunks for consistency
                    web_chunks = [
                        {
                            "id": f"web_{i}",
                            "content": result["content"],
                            "metadata": {
                                "source_file": result["url"],
                                "title": result["title"],
                                "search_type": "web"
                            }
                        }
                        for i, result in enumerate(web_results)
                    ]
                    
                    return {
                        "answer": web_llm_response.get("answer"),
                        "sources": web_llm_response.get("sources", []),
                        "chunks": web_chunks,
                        "context_chunks_used": web_llm_response.get("context_chunks_used", 0),
                        "model_used": web_llm_response.get("model_used", ""),
                        "search_type": "web_fallback_smart",
                        "error": web_llm_response.get("error"),
                        "original_document_answer": answer  # Keep the original answer for reference
                    }
                    
                except Exception as e:
                    # Return the original document-based answer since web search failed
                    return {
                        "answer": llm_response.get("answer"),
                        "sources": llm_response.get("sources", []),
                        "chunks": chunks,
                        "context_chunks_used": llm_response.get("context_chunks_used", 0),
                        "model_used": llm_response.get("model_used", ""),
                        "search_type": "document_web_error",
                        "error": f"Web search fallback error: {str(e)}"
                    }
            
            # Combine results (normal document search)
            return {
                "answer": llm_response.get("answer"),
                "sources": llm_response.get("sources", []),
                "chunks": chunks,  # Keep chunks for reference
                "context_chunks_used": llm_response.get("context_chunks_used", 0),
                "model_used": llm_response.get("model_used", ""),
                "search_type": "document",
                "error": llm_response.get("error")
            }
            
        except Exception as e:
            return {
                "error": f"Search failed: {str(e)}",
                "answer": None,
                "sources": [],
                "chunks": []
            }
