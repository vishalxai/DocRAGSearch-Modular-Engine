"""Graph builder module for the Self-Corrective RAG (CRAG) orchestrator."""

from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes.retrieval_node import RetrievalNode
from src.nodes.generation_node import GenerationNode
from src.nodes.web_search_node import WebSearchNode
from src.nodes.router import QuestionRouter
from src.nodes.grader_node import DocumentGrader # New

class RAGGraphBuilder:
    """Assembles the Agentic workflow with Self-Correction loops."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def build(self):
        """Compiles the Self-Corrective LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # 1. Instantiate the workers
        retrieve = RetrievalNode(self.retriever)
        generate = GenerationNode(self.llm)
        web_search = WebSearchNode()
        router = QuestionRouter(self.llm)
        grader = DocumentGrader(self.llm) # New: The Inspector

        # 2. Add the nodes to the graph
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_docs", grader) # New
        workflow.add_node("generate", generate)
        workflow.add_node("web_search", web_search)

        # 3. SET THE ENTRY POINT (Conditional)
        workflow.set_conditional_entry_point(
            router,
            {
                "vectorstore": "retrieve",
                "web_search": "web_search"
            }
        )

        # 4. DEFINE THE EDGES
        # Path A: Database Path
        workflow.add_edge("retrieve", "grade_docs")
        
        # Path B: The Decision Gate after Grading
        workflow.add_conditional_edges(
            "grade_docs",
            self.decide_to_generate,
            {
                "search": "web_search",
                "generate": "generate"
            }
        )

        # Path C: Web Path & Exit
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def decide_to_generate(self, state: GraphState):
        """
        Decision gate: Determines if we have enough info to answer 
        or if we need to fall back to the web.
        """
        print("--- DECIDING: GENERATE OR SEARCH? ---")
        # Check the flag we set in the Grader Node
        if state.get("answer") == "search_needed":
            print("--- DECISION: DOCUMENTS IRRELEVANT, ROUTING TO WEB SEARCH ---")
            return "search"
        else:
            print("--- DECISION: DOCUMENTS VALIDATED, PROCEEDING TO GENERATION ---")
            return "generate"