import streamlit as st
from pathlib import Path
from tempfile import TemporaryDirectory
from main import ResearchAssistant

# Configure Streamlit page
st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Research Assistant: PDF Document Manager")

# Initialize the research assistant
@st.cache_resource
def get_research_assistant():
    """Initialize and cache the ResearchAssistant instance"""
    return ResearchAssistant()

research_assistant = get_research_assistant()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload New Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        st.write("**Uploaded files:**")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")

        if st.button("Process and Index Files", type="primary"):
            with st.spinner("Processing files..."):
                with TemporaryDirectory() as tmpdir:
                    # Save uploaded files to temporary directory
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = Path(tmpdir) / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                        file_paths.append(file_path)

                    # Process files using the research assistant
                    try:
                        results = research_assistant.process_pdf_files(file_paths)
                        
                        # Display results
                        if results['processed_files']:
                            st.success(f"🎉 Successfully processed {len(results['processed_files'])} files!")
                            for file_info in results['processed_files']:
                                st.write(f"✅ {file_info['name']}: {file_info['chunks']} chunks indexed")
                        
                        if results['failed_files']:
                            st.warning(f"⚠️ {len(results['failed_files'])} files failed to process:")
                            for file_info in results['failed_files']:
                                st.write(f"❌ {file_info['name']}: {file_info['error']}")
                        
                        st.write(f"**Total chunks indexed:** {results['total_chunks']}")
                        
                        # Refresh the page to show new documents
                        # st.rerun()  # Removed to allow user to see results
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")

with col2:
    st.header("📋 Existing Documents")
    
    # Refresh button
    if st.button("🔄 Refresh", help="Reload documents from database"):
        st.rerun()
    
    # Get and display existing documents
    try:
        documents = research_assistant.get_all_documents()
        
        if documents:
            st.write(f"**Total documents: {len(documents)}**")
            
            # Delete all button with confirmation
            if st.button("🗑️ Delete All Documents", type="secondary", help="Delete all documents from the database"):
                if st.session_state.get('confirm_delete_all', False):
                    try:
                        deleted_count = research_assistant.delete_all_documents()
                        st.success(f"Deleted all {deleted_count} chunks from the database")
                        st.session_state['confirm_delete_all'] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting all documents: {str(e)}")
                else:
                    st.session_state['confirm_delete_all'] = True
                    st.warning("⚠️ Click again to confirm deletion of ALL documents")
            
            st.divider()
            
            # Display each document
            for source_file, doc_info in documents.items():
                with st.expander(f"📄 {source_file} ({doc_info['total_chunks']} chunks)", expanded=False):
                    
                    # Delete individual document button
                    col_info, col_delete = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**Chunks:** {doc_info['total_chunks']}")
                        
                    with col_delete:
                        if st.button(f"🗑️ Delete", key=f"delete_{source_file}", help=f"Delete {source_file}"):
                            try:
                                deleted_count = research_assistant.delete_document(source_file)
                                if deleted_count > 0:
                                    st.success(f"Deleted {deleted_count} chunks from {source_file}")
                                    st.rerun()
                                else:
                                    st.warning(f"No chunks found for {source_file}")
                            except Exception as e:
                                st.error(f"Error deleting document: {str(e)}")
                    
                    # Show first few chunks as preview
                    st.write("**Preview of chunks:**")
                    for i, chunk in enumerate(doc_info['chunks'][:3]):  # Show first 3 chunks
                        with st.container():
                            st.write(f"**Chunk {chunk['metadata'].get('chunk_index', i)}:**")
                            preview_text = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                            st.text(preview_text)
                            st.caption(f"Length: {chunk['metadata'].get('chunk_length', len(chunk['content']))} characters")
                    
                    if len(doc_info['chunks']) > 3:
                        st.caption(f"... and {len(doc_info['chunks']) - 3} more chunks")
        else:
            st.info("No documents found. Upload some PDFs to get started!")
            
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")

# Add some styling and information
st.divider()
st.markdown("""
### ℹ️ How to use:
1. **Upload PDFs**: Use the file uploader to select one or more PDF files
2. **Process**: Click "Process and Index Files" to extract text and create embeddings
3. **View**: See all your indexed documents in the "Existing Documents" section
4. **Delete**: Remove individual documents or all documents as needed
5. **Refresh**: Use the refresh button to reload the current state

**Note**: The app automatically refreshes after uploads and deletions to show the current state.

### 🏗️ Architecture:
- **UI Layer**: Streamlit handles all user interface components
- **Business Logic**: Main classes in `main.py` handle PDF processing, embeddings, and vector storage
- **Data Storage**: ChromaDB for persistent vector storage
- **AI Service**: OpenAI API for generating text embeddings
""")
