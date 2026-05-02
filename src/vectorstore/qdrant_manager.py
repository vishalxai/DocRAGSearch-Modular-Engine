import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

class QdrantManager:
    def __init__(self, collection_name="medical_docs"):
        """Initializes Qdrant Client and LangChain Vector Store"""
        self.collection_name = collection_name
        self.db_path = "./qdrant_db"

        # 1. Initialize local Qdrant Client
        print(f"Connecting to Qdrant at {self.db_path}...")
        self.client = QdrantClient(path=self.db_path)

        # 2. Ensure collection exists
        self._ensure_collection_exists()

        # 3. Setup embeddings via official OpenAI
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )

        # 4. Bind the LangChain wrapper to our Qdrant instance
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection_exists(self):
        """Creates collection if it doesn't exist"""
        if not self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print("Collection created successfully.")

    def add_documents(self, chunks):
        """Adds document chunks to Qdrant"""
        print(f"Adding {len(chunks)} chunks to Qdrant...")
        self.vector_store.add_documents(chunks)
        print("Documents added successfully.")

    def get_retriever(self, search_kwargs={"k": 5}):
        """Returns the retriever interface for LangGraph"""
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)