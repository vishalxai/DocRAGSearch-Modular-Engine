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
        self.prompt = PromptTemplate(
            template="""You are a strict, analytical assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If the answer is not contained within the context, you must strictly say "I do not know based on the provided documents."
            Do not use your pre-trained knowledge to answer.
            Keep the answer concise.

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
    
    