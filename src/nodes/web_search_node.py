"""Web Search Node for fetching real-time information via Tavily."""

from typing import Dict, Any
# FIX: The wrapper resides in the community utilities, not the native tavily package
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.documents import Document
from src.state import GraphState

class WebSearchNode:
    """Worker node responsible for querying the live internet."""
    
    def __init__(self):
        # We use the API Wrapper directly to get a clean List[Dict]
        self.search_wrapper = TavilySearchAPIWrapper()

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Executes the live web search."""
        print("\n--- NODE: WEB SEARCH ---")
        question = state["question"]
        print(f"Searching the live internet for: '{question}'")
        
        # 1. Use .results() to guarantee we get a List[Dict]
        # This prevents the 'string indices must be integers' error
        docs = self.search_wrapper.results(question, max_results=5)
        
        # 2. Format into Documents
        web_results = [
            Document(
                page_content=d["content"], 
                metadata={"source": d["url"]}
            ) for d in docs
        ]
        
        print(f"Found {len(web_results)} real-time web results.")
        
        # 3. Clear the 'answer' flag
        # This prevents the graph from getting confused if the grader ran before this
        return {"context": web_results, "answer": ""}