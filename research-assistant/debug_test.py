#!/usr/bin/env python3
"""
Debug script to test ResearchAssistant functionality
Run this to identify where the issue is occurring
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        from pypdf import PdfReader
        print("✅ pypdf import successful")
    except ImportError as e:
        print(f"❌ pypdf import failed: {e}")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ langchain import successful")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
    
    try:
        from openai import OpenAI
        print("✅ openai import successful")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
    
    try:
        import chromadb
        print("✅ chromadb import successful")
    except ImportError as e:
        print(f"❌ chromadb import failed: {e}")

def test_environment():
    """Test environment variables"""
    print("\nTesting environment...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY found (starts with: {api_key[:10]}...)")
    else:
        print("❌ OPENAI_API_KEY not found")

def test_research_assistant():
    """Test ResearchAssistant initialization"""
    print("\nTesting ResearchAssistant initialization...")
    try:
        from main import ResearchAssistant
        print("✅ ResearchAssistant import successful")
        
        assistant = ResearchAssistant()
        print("✅ ResearchAssistant initialization successful")
        
        # Test a simple search with empty database
        print("\nTesting search with empty database...")
        result = assistant.search_documents("test query")
        print(f"✅ Search completed. Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            if result.get("error"):
                print(f"Expected error (empty database): {result['error']}")
            elif result.get("answer"):
                print(f"Answer: {result['answer']}")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            
    except Exception as e:
        print(f"❌ ResearchAssistant test failed: {e}")

def test_individual_components():
    """Test individual components"""
    print("\nTesting individual components...")
    
    try:
        from main import PDFProcessor, EmbeddingService, VectorStore, LLMService
        
        # Test PDFProcessor
        pdf_processor = PDFProcessor()
        print("✅ PDFProcessor initialization successful")
        
        # Test EmbeddingService
        embedding_service = EmbeddingService()
        print("✅ EmbeddingService initialization successful")
        
        # Test VectorStore
        vector_store = VectorStore()
        print("✅ VectorStore initialization successful")
        
        # Test LLMService
        llm_service = LLMService()
        print("✅ LLMService initialization successful")
        
        # Test embedding generation
        if os.getenv("OPENAI_API_KEY"):
            print("\nTesting embedding generation...")
            try:
                embedding = embedding_service.get_embedding("test text")
                print(f"✅ Embedding generation successful. Length: {len(embedding)}")
            except Exception as e:
                print(f"❌ Embedding generation failed: {e}")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

if __name__ == "__main__":
    print("🔍 Research Assistant Debug Test")
    print("=" * 50)
    
    test_imports()
    test_environment()
    test_individual_components()
    test_research_assistant()
    
    print("\n" + "=" * 50)
    print("Debug test completed!") 