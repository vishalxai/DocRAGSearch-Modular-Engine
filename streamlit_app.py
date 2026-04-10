"""Streamlit interface for the DocRAG Enterprise Engine with File Uploads."""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Import our custom architecture
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.qdrant_manager import QdrantManager
from src.graph_builder.builder import RAGGraphBuilder

# 1. Page Configuration
st.set_page_config(
    page_title="DocRAG Enterprise Intelligence",
    page_icon="🧠",
    layout="wide" # Changed to wide to accommodate the sidebar
)

# 2. The Engine Cache 
@st.cache_resource
def load_systems():
    """Initializes the backend systems into persistent memory."""
    load_dotenv()
    
    # Init Processor & DB
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    db_manager = QdrantManager()
    
    # Init LLM via OpenRouter
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="openai/gpt-4o-mini", 
        temperature=0
    )
    
    # Build Graph Engine
    retriever = db_manager.get_retriever(search_kwargs={"k": 3})
    graph_builder = RAGGraphBuilder(retriever, llm)
    app = graph_builder.build()
    
    return processor, db_manager, app

# Load the engines
with st.spinner("Waking up the AI Engine..."):
    processor, db_manager, app = load_systems()

# ==========================================
# NEW: THE INGESTION SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📄 Knowledge Base")
    st.markdown("Upload PDFs or TXT files to add them to the AI's memory.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Ingest Documents") and uploaded_files:
        with st.spinner("Ingesting into Qdrant Vector DB..."):
            
            # Create a temporary directory to save the Streamlit RAM files to disk
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                
                # 1. Save uploaded files to the temp folder
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_file_path)
                
                # 2. Hand the file paths to our modular DocumentProcessor
                st.info("Extracting and Chunking text...")
                chunks = processor.process_sources(file_paths)
                
                # 3. Push the new chunks to Qdrant
                st.info(f"Embedding {len(chunks)} chunks...")
                db_manager.build_index(chunks)
                
            st.success("Ingestion Complete! The AI now knows this information.")

# ==========================================
# THE CHAT INTERFACE
# ==========================================
st.title("🧠 DocRAG Core v1.0")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Vector Database & Synthesizing..."):
            try:
                result = app.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Engine Error: {str(e)}")