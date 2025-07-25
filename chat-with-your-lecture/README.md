# 🎓 Lecture Chat Assistant

An interactive application that transforms lecture videos into a conversational experience using AI. Upload your lecture videos, get transcripts with embeddings, and chat with the content!

## 🚀 Features

- **Video Processing**: Upload MP4, MOV, or AVI lecture videos
- **Audio Extraction**: Automatic audio extraction using FFmpeg
- **Speech-to-Text**: High-quality transcription using OpenAI Whisper
- **Smart Chunking**: Intelligent text segmentation with timestamp preservation
- **Vector Search**: Semantic search using sentence transformers and ChromaDB
- **AI Chat**: Interactive Q&A powered by OpenAI GPT models
- **Tabbed Interface**: Clean, organized UI with separate upload and chat phases
- **Progress Tracking**: Real-time progress indicators during processing
- **Timestamp References**: Answers include relevant video timestamps

## 📋 Implementation Flow

1. **Upload video or audio** - Upload lecture content through the web interface
2. **Extract transcript** - Audio extraction and Whisper-based transcription
3. **Convert to embeddings** - Transform text chunks into vector embeddings
4. **Save to vector db** - Store embeddings in ChromaDB for fast retrieval
5. **User ask query** - Interactive chat interface for questions
6. **Convert query to embedding** - Process questions using same embedding model
7. **Find top k in vector db** - Semantic search for relevant content
8. **Send query and context to LLM** - Generate answers using OpenAI GPT
9. **Get answer and show on UI** - Display responses with timestamps

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install streamlit whisper ffmpeg-python sentence-transformers chromadb openai python-dotenv
```

### 2. Set Up Environment

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. Get your OpenAI API key from: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
streamlit run app.py
```

## 💾 Embedding Storage

The application uses **persistent ChromaDB storage** to save your embeddings:

- **Location**: `./chroma_db/` directory in your project folder
- **Persistence**: Embeddings are saved to disk and persist between sessions
- **Collections**: Each video gets a unique collection name based on filename and content
- **Automatic**: Storage happens automatically during video processing

**Note**: The `chroma_db/` directory will be created automatically when you process your first video.

## 🎯 How to Use

### Phase 1: Upload & Process (📹 Tab)
1. Navigate to the "Upload & Process" tab
2. Click "Choose a lecture video" and select your file
3. Watch the progress bar as the system:
   - Extracts audio from your video
   - Generates transcript using Whisper
   - Creates vector embeddings
   - Stores content in the database
4. Review the processing summary (chunks, duration, word count)

### Phase 2: Chat with Lecture (💬 Tab)
1. Switch to the "Chat with Lecture" tab
2. Type your questions about the lecture content
3. Get AI-powered answers with relevant timestamps
4. Use sample questions for inspiration
5. View timestamp references in expandable sections

## 🎨 UI Features

- **Wide Layout**: Optimized for desktop viewing
- **Progress Tracking**: Visual feedback during video processing
- **Metrics Display**: Processing summary with key statistics
- **Interactive Chat**: Clean question/answer interface
- **Sample Questions**: Pre-built queries to get started
- **Timestamp Integration**: Easy navigation to relevant video sections
- **Error Handling**: User-friendly error messages and guidance

## 🔧 Technical Details

- **Whisper Model**: Base model for speech recognition
- **Embedding Model**: all-MiniLM-L6-v2 for semantic search
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM**: OpenAI GPT-4o-mini for answer generation
- **Chunk Size**: 500 characters max per text chunk
- **Search Results**: Top 3 most relevant chunks per query

## 📁 Project Structure

```
chat-with-your-lecture/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## 🚨 Troubleshooting

### SSL Certificate Issues (macOS)
The app includes automatic SSL context handling for macOS certificate issues.

### Missing Dependencies
Install missing packages using pip:
```bash
pip install -r requirements.txt
```

### OpenAI API Errors
- Ensure your API key is correctly set in `.env`
- Check your OpenAI account has sufficient credits
- Verify the API key has the correct permissions

### Memory Issues
For large videos, ensure sufficient system memory. Consider using smaller video files or cloud deployment for heavy processing.

## 🎓 Sample Use Cases

- **Students**: Review lecture content, clarify concepts, find specific topics
- **Educators**: Create interactive study materials from recorded lectures
- **Researchers**: Extract insights from conference presentations
- **Professionals**: Make training videos searchable and interactive