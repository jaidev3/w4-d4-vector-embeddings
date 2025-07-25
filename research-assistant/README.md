# Research Assistant: PDF Document Manager

A Streamlit-based application for managing and indexing PDF documents using vector embeddings and ChromaDB for semantic search capabilities.

## 🏗️ Architecture

The application follows a clean separation of concerns:

### UI Layer (`streamlit_app.py`)
- Handles all user interface components using Streamlit
- File upload interface
- Document management dashboard  
- Progress indicators and user feedback
- Error handling and display

### Business Logic Layer (`main.py`)
Contains four main classes:

#### `PDFProcessor`
- Extracts text from PDF files using `pypdf`
- Splits text into chunks using LangChain's `RecursiveCharacterTextSplitter`
- Configurable chunk size and overlap

#### `EmbeddingService`  
- Generates text embeddings using OpenAI's API
- Uses `text-embedding-3-small` model by default
- Handles API authentication and rate limiting

#### `VectorStore`
- Manages ChromaDB operations for persistent vector storage
- Handles document CRUD operations
- Groups documents by source file for easy management

#### `ResearchAssistant`
- Main orchestrator class that coordinates all operations
- Processes multiple PDF files in batch
- Provides high-level API for the UI layer

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📁 File Structure

```
research-assistant/
├── main.py              # Business logic classes
├── streamlit_app.py     # Streamlit UI
├── requirements.txt     # Python dependencies
├── chroma_db/          # ChromaDB persistent storage
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_EMBEDDING_MODEL`: Embedding model to use (optional, defaults to `text-embedding-3-small`)

### ChromaDB Settings
- Default persist directory: `chroma_db/`
- Default collection name: `research_papers`
- These can be customized in the `VectorStore` class initialization

### Text Processing Settings
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Separators: `["\n\n", "\n", " "]`

## 📚 Usage

1. **Upload PDFs**: Use the file uploader to select one or more PDF files
2. **Process**: Click "Process and Index Files" to extract text and create embeddings
3. **View**: See all your indexed documents in the "Existing Documents" section
4. **Delete**: Remove individual documents or all documents as needed
5. **Refresh**: Use the refresh button to reload the current state

## 🔍 Features

- **Multi-file Processing**: Upload and process multiple PDFs simultaneously
- **Vector Embeddings**: Uses OpenAI's latest embedding models for semantic search
- **Persistent Storage**: ChromaDB ensures your data persists between sessions
- **Document Management**: Easy viewing, deletion, and organization of indexed documents
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Progress Tracking**: Real-time feedback during processing operations

## 🛠️ Extending the Application

The modular architecture makes it easy to extend:

- **Add new document types**: Extend `PDFProcessor` or create new processor classes
- **Different embedding providers**: Modify `EmbeddingService` to use other APIs
- **Search functionality**: Implement the `search_documents` method in `ResearchAssistant`
- **Custom storage**: Replace `VectorStore` with different vector databases

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **OpenAI API Errors**: Verify your API key is correct and has sufficient credits
3. **ChromaDB Issues**: Check write permissions for the `chroma_db/` directory
4. **Memory Issues**: For large PDFs, consider reducing chunk size or processing fewer files at once

### Linter Warnings
The linter may show import warnings if packages aren't installed in your development environment. These are expected and won't affect runtime when dependencies are properly installed. 