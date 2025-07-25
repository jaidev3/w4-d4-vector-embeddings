# Research Assistant: PDF Document Manager with AI-Powered Q&A

A Streamlit-based application that allows you to upload PDF documents, create vector embeddings, and ask questions with AI-powered answers using Retrieval-Augmented Generation (RAG).

## 🚀 Features

### Document Management
- **PDF Upload**: Upload multiple PDF files simultaneously
- **Text Extraction**: Automatic text extraction from PDF documents
- **Chunking**: Intelligent text splitting for optimal embedding generation
- **Vector Storage**: Persistent storage using ChromaDB
- **Document Management**: View, delete individual documents or clear all documents

### AI-Powered Q&A System
- **Smart Search**: Convert queries to embeddings and find semantically similar content
- **Web Search Fallback**: Automatically search the web using Tavily when no relevant documents are found
- **LLM Integration**: Generate comprehensive answers using OpenAI's GPT-4o-mini
- **Source Attribution**: Track which documents or web sources contributed to each answer
- **Configurable Retrieval**: Choose how many document chunks to analyze (3, 5, or 10)
- **Transparent Process**: View the exact document chunks or web results used for answer generation
- **Hybrid Search**: Seamlessly combines local document knowledge with real-time web information

## 🏗️ Architecture

The application follows a modular Hybrid RAG (Retrieval-Augmented Generation) architecture with web search fallback:

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
                                ↓ (if no results)
                           Web Search (Tavily) → Web Context → LLM → Answer
```

### Core Components

1. **PDFProcessor**: Handles PDF text extraction and chunking
2. **EmbeddingService**: Generates OpenAI embeddings for text
3. **WebSearchService**: Handles web search using Tavily API as fallback
4. **LLMService**: Generates answers using OpenAI chat completions (supports both document and web contexts)
5. **VectorStore**: Manages ChromaDB operations for document storage and retrieval
6. **ResearchAssistant**: Orchestrates the entire hybrid search pipeline

## 🛠️ Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   
   **Note**: The Tavily API key is optional. If not provided, the system will still work but web search fallback will be disabled.

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📖 Usage

### 1. Upload Documents
- Use the file uploader to select PDF files
- Click "Process and Index Files" to extract text and create embeddings
- Monitor the processing results and chunk counts

### 2. Ask Questions
- Type your question in the Q&A section
- Select how many document chunks to analyze (3, 5, or 10)
- Click "Get Answer" to receive an AI-generated response
- View source attribution and retrieved chunks

### 3. Manage Documents
- View all indexed documents with chunk counts
- Delete individual documents or all documents
- Refresh to see current database state

## 🔧 Configuration

### Embedding Model
- Default: `text-embedding-3-small`
- Configurable in `EmbeddingService`

### LLM Model
- Default: `gpt-3.5-turbo`
- Configurable in `LLMService.generate_answer()`

### Text Chunking
- Chunk size: 1000 characters
- Overlap: 200 characters
- Configurable in `PDFProcessor`

## 🎯 Example Queries

- "What are the main topics covered in these documents?"
- "Summarize the key findings from the research papers"
- "What methodology was used in the studies?"
- "What are the conclusions and recommendations?"

## 🔍 Technical Details

### Hybrid RAG Pipeline
1. **Document Ingestion**: PDFs → Text Extraction → Chunking
2. **Embedding Generation**: Text Chunks → OpenAI Embeddings → Vector Storage
3. **Query Processing**: User Query → Query Embedding → Similarity Search
4. **Context Selection**: 
   - If documents found: Retrieved Chunks → LLM Context
   - If no documents: Web Search (Tavily) → Web Results → LLM Context
5. **Answer Generation**: Context → LLM → Generated Answer with Source Attribution

### Data Flow
```
PDF Files → Text Chunks → Embeddings → ChromaDB
                                         ↓
User Query → Query Embedding → Vector Search → Top-K Chunks → LLM → Answer
                                         ↓ (if no results)
                                    Web Search (Tavily) → Web Results → LLM → Answer
```

## 🌐 Web Search Fallback

When the system cannot find relevant information in your uploaded documents, it automatically falls back to web search using the Tavily API. This ensures you always get comprehensive answers, whether from your documents or current web information.

### How it works:
1. **Primary Search**: Query your uploaded documents first
2. **Relevance Detection**: LLM analyzes if documents contain relevant information
3. **Smart Fallback Trigger**: If documents don't contain relevant info OR no documents found
4. **Web Search**: Automatically search the web using Tavily for current information
5. **Answer Generation**: Use web results to generate comprehensive answers
6. **Source Attribution**: Clearly indicate whether answers come from documents or web search

### Benefits:
- **Always Get Answers**: Never get stuck with "no results found"
- **Smart Relevance Detection**: Automatically detects when documents don't contain relevant information
- **Current Information**: Access to real-time web information for topics like weather, news, stock prices
- **Transparent Sources**: Clear indication of whether information comes from your documents or the web
- **Seamless Experience**: Automatic fallback without user intervention
- **Best of Both Worlds**: Combines your private document knowledge with current web information

## 📋 Dependencies

- `streamlit`: Web application framework
- `openai`: OpenAI API client for embeddings and chat completions
- `chromadb`: Vector database for semantic search
- `pypdf`: PDF text extraction
- `langchain`: Text splitting utilities
- `langchain-tavily`: Tavily search integration for web search fallback
- `python-dotenv`: Environment variable management

## 🚨 Notes

- Requires OpenAI API key for both embeddings and chat completions
- ChromaDB data persists locally in the `chroma_db` directory
- Processing time depends on document size and number of chunks
- LLM responses are generated based solely on uploaded document content 