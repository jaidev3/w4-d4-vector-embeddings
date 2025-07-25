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
- **LLM Integration**: Generate comprehensive answers using OpenAI's GPT-3.5-turbo
- **Source Attribution**: Track which documents contributed to each answer
- **Configurable Retrieval**: Choose how many document chunks to analyze (3, 5, or 10)
- **Transparent Process**: View the exact document chunks used for answer generation

## 🏗️ Architecture

The application follows a modular RAG (Retrieval-Augmented Generation) architecture:

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
```

### Core Components

1. **PDFProcessor**: Handles PDF text extraction and chunking
2. **EmbeddingService**: Generates OpenAI embeddings for text
3. **LLMService**: Generates answers using OpenAI chat completions
4. **VectorStore**: Manages ChromaDB operations for document storage and retrieval
5. **ResearchAssistant**: Orchestrates the entire pipeline

## 🛠️ Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

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

### RAG Pipeline
1. **Document Ingestion**: PDFs → Text Extraction → Chunking
2. **Embedding Generation**: Text Chunks → OpenAI Embeddings → Vector Storage
3. **Query Processing**: User Query → Query Embedding → Similarity Search
4. **Answer Generation**: Retrieved Chunks → LLM Context → Generated Answer

### Data Flow
```
PDF Files → Text Chunks → Embeddings → ChromaDB
                                         ↓
User Query → Query Embedding → Vector Search → Top-K Chunks → LLM → Answer
```

## 📋 Dependencies

- `streamlit`: Web application framework
- `openai`: OpenAI API client for embeddings and chat completions
- `chromadb`: Vector database for semantic search
- `pypdf`: PDF text extraction
- `langchain`: Text splitting utilities
- `python-dotenv`: Environment variable management

## 🚨 Notes

- Requires OpenAI API key for both embeddings and chat completions
- ChromaDB data persists locally in the `chroma_db` directory
- Processing time depends on document size and number of chunks
- LLM responses are generated based solely on uploaded document content 