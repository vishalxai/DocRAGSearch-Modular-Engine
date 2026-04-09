"""Document processing module for loading and splitting documents."""

from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Modern, non-deprecated import

class DocumentProcessor:
    """Handles document loading and processing for Enterprise RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Increased chunk size from 500 to 1000 to retain better LLM context
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL."""
        print(f"Scraping URL: {url}")
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory."""
        print(f"Scanning directory: {directory}")
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file."""
        print(f"Loading TXT: {file_path}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a SINGLE PDF file."""
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(str(file_path)) 
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """Smart router: Loads from URLs, directories, or specific files."""
        docs: List[Document] = []
        
        for src in sources:
            # 1. Handle Web URLs
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue 
           
            # 2. Handle Local Files / Directories
            path = Path(src) 
            
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.is_file():
                if path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))
                elif path.suffix.lower() == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                else:
                    print(f"Warning: Skipping unsupported file type -> {src}")
            else:
                print(f"Warning: Source not found -> {src}")
                
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        print(f"Splitting {len(documents)} document(s)...")
        return self.splitter.split_documents(documents)
    
    def process_sources(self, sources: List[str]) -> List[Document]: 
        """Complete pipeline: Load -> Split."""
        docs = self.load_documents(sources)
        chunks = self.split_documents(docs)
        print(f"Pipeline complete: Generated {len(chunks)} chunks.")
        return chunks