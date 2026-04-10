"""Graph builder module for assembling the LangGraph orchestrator."""

from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes.retrieval_node import RetrievalNode
from src.nodes.generation_node import GenerationNode

class RAGGraphBuilder:
    """Assembles the retrieval and generation nodes into a state machine."""

    def __init__(self, retriever, llm):
        """Pass in the database connection and the LLM engine."""
        self.retriever = retriever
        self.llm = llm

    def build(self):
        """Compiles the LangGraph workflow."""
        # 1. Initialize the Graph with our specific clipboard blueprint
        workflow = StateGraph(GraphState)

        # 2. Instantiate the workers
        retrieve = RetrievalNode(self.retriever)
        generate = GenerationNode(self.llm)

        # 3. Add the workers to the factory floor (Nodes)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)

        # 4. Lay down the conveyor belts (Edges)
        workflow.set_entry_point("retrieve")          # Start here
        workflow.add_edge("retrieve", "generate")     # Pass from retrieve to generate
        workflow.add_edge("generate", END)            # Finish the job

        # 5. Compile the machine into an executable application
        return workflow.compile()