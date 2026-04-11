"""Supervisor logic to direct questions to the optimal data source."""

from typing import Literal
from langchain_core.prompts import PromptTemplate

class QuestionRouter:
    """Agentic router that decides whether to use RAG or Web Search."""
    
    def __init__(self, llm):
        self.llm = llm
        
        # The prompt that forces the LLM to make a binary routing decision
        self.prompt = PromptTemplate(
            template="""You are an expert AI architecture router. 
            Your job is to direct a user's question to either a local vector database or a live web search.
            
            The local database contains documents about: Artificial Intelligence, Machine Learning, and technical interview questions.
            If the question is related to those topics or implies reading an uploaded document, route to 'vectorstore'.
            If the question asks for current events, real-time weather, or general knowledge outside those bounds, route to 'web_search'.
            
            You must return ONLY the exact string 'vectorstore' or 'web_search'. No other text, no punctuation.
            
            Question: {question}
            Decision:""",
            input_variables=["question"]
        )
        self.chain = self.prompt | self.llm

    def __call__(self, state) -> Literal["vectorstore", "web_search"]:
        """Evaluates the state and returns the routing path."""
        print("\n--- NODE: SUPERVISOR ROUTER ---")
        question = state["question"]
        print(f"Analyzing question intent...")
        
        # Ask the LLM to make the decision
        response = self.chain.invoke({"question": question})
        decision = response.content.strip().lower()
        
        # Fallback logic just in case the LLM includes extra punctuation
        if "web_search" in decision:
            print("Decision: Route to LIVE WEB SEARCH")
            return "web_search"
        else:
            print("Decision: Route to LOCAL VECTOR DATABASE")
            return "vectorstore"