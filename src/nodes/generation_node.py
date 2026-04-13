"""Generation Node for synthesizing the final answer using the LLM."""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from src.state import GraphState

class GenerationNode:
    """
    Worker node responsible for taking the retrieved context and user question,
    and generating a final answer using an LLM.
    """

    def __init__(self, llm):
        """Initialize with a LangChain LLM (ChatOpenAI) object."""
        self.llm = llm

        # 1. Define the Enterprise Guardrail Prompt
        # Added smarter logic to distinguish between Web and DB context
        self.prompt = PromptTemplate(
            template="""You are the internal 'Apex AI Labs' Engineering Copilot. 
            Your SOLE purpose is to assist software and machine learning engineers with technical problems, coding, and internal documentation.

            CRITICAL GUARDRAIL: If the user asks a question completely unrelated to Software Engineering, AI, ML, or Data Science (e.g., cooking, sports, creative writing, general trivia), you must immediately reply EXACTLY with:
            "SECURITY REFUSAL: Query outside of enterprise domain parameters. I am authorized only for technical engineering assistance."

            If the query is within the technical domain:
            - Use the retrieved context to answer.
            - If the context is from a Web Search, provide a helpful technical summary.
            - If the context is from the Document Database, be extremely precise.
            - If the context is empty but the query is technical, say "I do not have enough internal documentation to answer this."

            Question: {question} 
            Context: {context} 
            Answer:""",
            input_variables=["question", "context"],
        )
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        The actual execution logic of the node.
        
        Args:
            state (GraphState): The current clipboard containing 'question' and 'context'.
            
        Returns:
            Dict: A dictionary with the final 'answer' to update the GraphState.
        """
        print("\n--- NODE: GENERATION ---")
        
        # 2. Read from the clipboard
        question = state["question"]
        documents = state["context"]
        
        # 3. Format the context (Convert list of Document objects into a single string)
        context_str = "\n\n".join([doc.page_content for doc in documents])
        print(f"Feeding {len(documents)} context chunks to the LLM...")
        
        # 4. Chain the prompt and the LLM together
        chain = self.prompt | self.llm
        
        # 5. Execute the LLM call
        response = chain.invoke({"context": context_str, "question": question})
        
        print("Answer generated successfully.")
        
        # 6. Write the final answer back to the clipboard
        return {"answer": response.content}