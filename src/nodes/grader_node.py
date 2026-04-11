"""Grader Node for verifying document relevance."""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from src.state import GraphState

class DocumentGrader:
    """Worker node that evaluates the quality of retrieved context."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
            It does not need to be a complete answer, just a useful piece of information.
            
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
            Return ONLY the string 'yes' or 'no'.

            Retrieved Document: {context}
            User Question: {question}
            
            Score:""",
            input_variables=["context", "question"],
        )
        self.chain = self.prompt | self.llm

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- NODE: GRADING DOCUMENTS ---")
        question = state["question"]
        documents = state["context"]
        
        filtered_docs = []
        search_needed = False
        
        for doc in documents:
            score = self.chain.invoke({"question": question, "context": doc.page_content})
            grade = score.content.strip().lower()
            
            if "yes" in grade:
                print("--- GRADE: DOCUMENT RELEVANT ---")
                filtered_docs.append(doc)
            else:
                print("--- GRADE: DOCUMENT NOT RELEVANT ---")
                # If we find even one irrelevant document, we flag that we might need web search
                search_needed = True
        
        # If we have NO relevant documents left, we MUST trigger web search
        if not filtered_docs:
            print("--- DECISION: ALL DOCUMENTS IRRELEVANT, TRIGGERING WEB SEARCH ---")
            return {"context": [], "answer": "search_needed"} 
            
        return {"context": filtered_docs}