"""Retrieval Node for fetching context from the Vector Store."""

from typing import Dict, Any
from src.state import GraphState

class RetrievalNode:
    """
    Worker node responsible for querying Qdrant and updating the state context.
    We use a class here so we can pass the database connection in once during setup.
    """
    
    def __init__(self, retriever):
        """Initialize with a LangChain retriever object."""
        self.retriever = retriever

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        The actual execution logic of the node.
        
        Args:
            state (GraphState): The current clipboard containing the user's question.
            
        Returns:
            Dict: A dictionary with the exact key we want to update in the GraphState.
        """
        print("\n--- NODE: RETRIEVAL ---")
        
        # 1. Read the question from the clipboard
        question = state["question"]
        print(f"Searching Qdrant for: '{question}'")
        
        # 2. Execute the similarity search against the Vector DB
        documents = self.retriever.invoke(question)
        
        print(f"Found {len(documents)} relevant chunks of context.")
        
        # 3. Write the results back to the clipboard
        # LangGraph automatically takes this dictionary and overwrites the 'context' variable in GraphState.
        return {"context": documents}