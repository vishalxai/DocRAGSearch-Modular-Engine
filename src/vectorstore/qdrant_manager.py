"""Vector Store module for managing Qdrant database operations."""
import os

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams # Added to define schema

class QdrantManager:
    """Handles embedding and persistent storage of document chunks."""
    
    def __init__(self, collection_name: str = "doc_rag_core", db_path: str = "./qdrant_db"):
        self.collection_name = collection_name
        self.db_path = db_path
        
        # 1. Initialize the Embedding Model (Routed through OpenRouter)
        print("Initializing Embeddings via OpenRouter...")
        self.embeddings = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            # OpenRouter requires the 'openai/' prefix to route correctly
            model="openai/text-embedding-3-small" 
        )
        
        # 2. Connect to the Qdrant Database 
        print(f"Connecting to Qdrant at {self.db_path}...")
        self.client = QdrantClient(path=self.db_path)

        # 3. CRITICAL NEW STEP: Ensure the Collection exists before LangChain binds to it
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating new Qdrant collection: '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536, # OpenAI Embeddings are exactly 1536 dimensions
                    distance=Distance.COSINE # Standard distance metric for text similarity
                ),
            )
        
        # 4. Bind the LangChain wrapper to our Qdrant instance
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def build_index(self, chunks: List[Document]):
        """Ingest document chunks into the Qdrant database."""
        if not chunks:
            print("Warning: No chunks provided. Skipping ingestion.")
            return
            
        print(f"Pushing {len(chunks)} chunks to collection: '{self.collection_name}'...")
        self.vector_store.add_documents(chunks)
        print("Indexing complete. Data is now persistent.")

    def get_retriever(self, search_kwargs: dict = {"k": 4}):
        """Returns a retriever object for LangGraph to use later."""
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)