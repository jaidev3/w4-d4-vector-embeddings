import streamlit as st
import whisper
import os
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import ffmpeg
import ssl
import urllib.request
from openai import OpenAI
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix SSL certificate verification issues on macOS
def fix_ssl_context():
    """Handle SSL certificate issues for model downloads"""
    try:
        # First, try to use the system certificates
        import certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # If certifi is not available, create unverified context as fallback
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        st.warning("Using unverified SSL context. Consider installing certifi for better security.")
    except Exception:
        # Last resort - unverified context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        # Install the context as the default
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
        )
    except Exception as e:
        st.warning(f"Could not configure SSL context: {e}")

# Initialize models and clients
@st.cache_resource
def load_models():
    # Fix SSL issues before loading models
    fix_ssl_context()
    
    try:
        whisper_model = whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.error("Please check your internet connection and SSL certificates")
        return None, None, None
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create persistent ChromaDB client with local storage
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    return whisper_model, embedding_model, chroma_client

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream)
    except ffmpeg.Error as e:
        st.error(f"Error extracting audio: {e}")
        return False
    return True

# Function to transcribe audio with timestamps
def transcribe_audio(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    segments = []
    for segment in result['segments']:
        segments.append({
            'text': segment['text'],
            'start': segment['start'],
            'end': segment['end']
        })
    return segments

# Function to chunk transcript with timestamps
def chunk_transcript(segments, max_chunk_size=500):
    chunks = []
    current_chunk = ""
    chunk_start = None
    chunk_end = None
    
    for segment in segments:
        segment_text = segment['text']
        segment_start = segment['start']
        segment_end = segment['end']
        
        if len(current_chunk) + len(segment_text) <= max_chunk_size:
            if not current_chunk:
                chunk_start = segment_start
            current_chunk += " " + segment_text
            chunk_end = segment_end
        else:
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start': chunk_start,
                    'end': chunk_end
                })
            current_chunk = segment_text
            chunk_start = segment_start
            chunk_end = segment_end
    
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'start': chunk_start,
            'end': chunk_end
        })
    
    return chunks

# Function to format timestamp
def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Lecture Chat Assistant",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Lecture Chat Assistant")
    st.markdown("Transform your lecture videos into an interactive chat experience")
    
    # Initialize session state
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'video_name' not in st.session_state:
        st.session_state.video_name = ""
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = ""
    
    # Load models
    whisper_model, embedding_model, chroma_client = load_models()
    
    # Check if models loaded successfully
    if whisper_model is None or embedding_model is None or chroma_client is None:
        st.error("❌ Failed to load required models. Please check your internet connection and try again.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📹 Upload & Process", "💬 Chat with Lecture"])
    
    with tab1:
        st.header("📹 Upload & Process Your Lecture")
        
        # Show current status
        if st.session_state.processing_complete:
            st.success(f"✅ Lecture '{st.session_state.video_name}' has been processed successfully!")
            st.info("📝 You can now switch to the 'Chat with Lecture' tab to start asking questions.")
            
            # Option to process a new video
            if st.button("🔄 Process New Video", type="secondary"):
                st.session_state.processing_complete = False
                st.session_state.transcripts = []
                st.session_state.vector_db = None
                st.session_state.video_name = ""
                st.session_state.collection_name = ""
                st.rerun()
        else:
            st.markdown("""
            **Instructions:**
            1. Upload your lecture video (MP4, MOV, or AVI format)
            2. Wait for the system to extract audio and generate transcript
            3. The content will be processed and stored for querying
            4. Switch to the Chat tab once processing is complete
            """)
            
            # Video upload
            uploaded_file = st.file_uploader(
                "Choose a lecture video", 
                type=['mp4', 'mov', 'avi'],
                help="Supported formats: MP4, MOV, AVI. Max file size depends on your system memory."
            )
            
            if uploaded_file is not None:
                st.session_state.video_name = uploaded_file.name
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(uploaded_file.read())
                    video_path = tmp_video.name
                
                progress_bar.progress(20)
                status_text.text("🎵 Extracting audio from video...")
                
                # Extract audio
                audio_path = video_path.replace('.mp4', '.wav')
                if extract_audio(video_path, audio_path):
                    progress_bar.progress(40)
                    status_text.text("✅ Audio extracted successfully")
                    
                    # Transcribe audio
                    status_text.text("🎤 Generating transcript with Whisper...")
                    segments = transcribe_audio(audio_path, whisper_model)
                    chunks = chunk_transcript(segments)
                    st.session_state.transcripts = chunks
                    
                    progress_bar.progress(70)
                    status_text.text("📝 Transcript generated, creating embeddings...")
                    
                    # Create vector database with unique collection name for each video
                    collection_name = f"lecture_{hash(uploaded_file.name)}_{len(chunks)}"
                    
                    # Check if collection exists and delete if it does
                    existing_collections = [c.name for c in chroma_client.list_collections()]
                    if collection_name in existing_collections:
                        chroma_client.delete_collection(name=collection_name)
                    
                    # Create new collection
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name="all-MiniLM-L6-v2"
                        )
                    )
                    
                    # Store chunks in vector database
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            documents=[chunk['text']],
                            metadatas=[{
                                'start': chunk['start'],
                                'end': chunk['end']
                            }],
                            ids=[f"chunk_{i}"]
                        )
                    st.session_state.vector_db = collection
                    st.session_state.collection_name = collection_name
                    
                    progress_bar.progress(100)
                    status_text.text("🎉 Processing complete!")
                    st.session_state.processing_complete = True
                    
                    # Show processing summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Chunks", len(chunks))
                    with col2:
                        total_duration = max([chunk['end'] for chunk in chunks])
                        st.metric("⏱️ Video Duration", format_timestamp(total_duration))
                    with col3:
                        total_words = sum([len(chunk['text'].split()) for chunk in chunks])
                        st.metric("📝 Total Words", total_words)
                    
                    # Clean up temporary files
                    os.unlink(video_path)
                    os.unlink(audio_path)
                    
                    st.success("✅ Your lecture has been processed! Switch to the 'Chat with Lecture' tab to start asking questions.")
                    
                    # Show storage information
                    with st.expander("📁 Storage Information", expanded=False):
                        persist_dir = os.path.join(os.getcwd(), "chroma_db")
                        st.info(f"**Embeddings saved to:** `{persist_dir}`")
                        st.info(f"**Collection name:** `{collection_name}`")
                        st.info("Embeddings are now persistently stored and will be available in future sessions.")
                else:
                    st.error("❌ Failed to extract audio from video. Please try a different file.")
    
    with tab2:
        st.header("💬 Chat with Your Lecture")
        
        if not st.session_state.processing_complete:
            st.warning("⚠️ Please upload and process a lecture video in the 'Upload & Process' tab first.")
            st.markdown("""
            **To get started:**
            1. Go to the 'Upload & Process' tab
            2. Upload your lecture video
            3. Wait for processing to complete
            4. Return here to start chatting!
            """)
        else:
            # Show lecture info
            st.info(f"📹 Currently chatting with: **{st.session_state.video_name}**")
            
            # Show available collections for debugging
            if st.checkbox("🔍 Show Debug Info", value=False):
                try:
                    all_collections = chroma_client.list_collections()
                    st.write(f"**Available collections:** {[c.name for c in all_collections]}")
                    st.write(f"**Current collection:** {st.session_state.collection_name}")
                except Exception as e:
                    st.write(f"Error listing collections: {e}")
            
            # Chat interface with better styling
            st.markdown("### 🤔 Ask questions about your lecture:")
            
            # Create columns for better layout
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_query = st.text_input(
                    "Your question:",
                    placeholder="e.g., What are the main topics covered? Can you summarize the key points?",
                    label_visibility="collapsed"
                )
            
            with col2:
                ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            # Process query
            if user_query and (ask_button or user_query):
                with st.spinner("🔍 Searching through lecture content..."):
                    # Convert query to embedding
                    query_embedding = embedding_model.encode([user_query])[0]
                    
                    # Retrieve relevant chunks
                    results = st.session_state.vector_db.query(
                        query_embeddings=[query_embedding],
                        n_results=3
                    )
                    
                    # Prepare context for LLM
                    context = ""
                    timestamps = []
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        context += f"{doc}\n"
                        timestamps.append(f"{format_timestamp(metadata['start'])} - {format_timestamp(metadata['end'])}")
                    
                    # Generate response using OpenAI
                    try:
                        # Initialize OpenAI client with API key from environment
                        api_key = os.getenv('OPENAI_API_KEY')
                        if not api_key:
                            st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
                            return
                        
                        openai_client = OpenAI(api_key=api_key)
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant answering questions about lecture content. Provide accurate, detailed answers based on the given context. Include relevant timestamp references when helpful. Format your response in a clear, structured way."},
                                {"role": "user", "content": f"Context from lecture: {context}\n\nQuestion: {user_query}"}
                            ]
                        )
                        
                        answer = response.choices[0].message.content
                        
                        # Display answer with better formatting
                        st.markdown("### 💡 Answer:")
                        st.markdown(answer)
                        
                        # Show relevant timestamps in an expandable section
                        with st.expander("🕒 Relevant Timestamps", expanded=False):
                            st.markdown("These sections of the lecture are most relevant to your question:")
                            for i, ts in enumerate(timestamps, 1):
                                st.markdown(f"**Section {i}:** {ts}")
                        
                        # Add some spacing
                        st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")
            
            # Add sample questions
            st.markdown("### 💭 Sample Questions:")
            sample_questions = [
                "What are the main topics covered in this lecture?",
                "Can you summarize the key points?",
                "What examples were given?",
                "What are the important definitions mentioned?",
                "What homework or assignments were discussed?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(f"💬 {question}", key=f"sample_{i}", use_container_width=True):
                        st.session_state.sample_question = question
                        # This will be handled by the query processing above

if __name__ == "__main__":
    main()