# DocRAGSearch Modular Engine: Enterprise Edition 🏢🧠

An Enterprise-grade, Agentic Retrieval-Augmented Generation (RAG) engine built with LangGraph, Qdrant, and Docker. 

Unlike standard "Chat with PDF" wrappers that blindly retrieve context or hallucinate when data is missing, this system implements **Corrective RAG (CRAG)** and strict **Domain Guardrails**. It is designed to prevent LLM hallucinations, mitigate "Agentic Drift," and eliminate wasted API costs by actively rejecting non-business queries.

## 🏗 Architecture & Core Features

* **Agentic Routing (LangGraph):** Dynamically routes user queries to the local vector database or the live internet based on context and intent.
* **Self-Correcting Loop (CRAG):** Features a dedicated "Grader Node." If local database chunks are deemed irrelevant to the query, the system safely overrides retrieval and falls back to a live web search.
* **Enterprise Guardrails:** Enforces a strict technical domain (Software Engineering, Data Science, and Machine Learning). It aggressively rejects off-topic queries (e.g., "baking a cake", "writing a poem") to save API tokens and maintain corporate compliance.
* **Stateless Containerization:** Fully Dockerized using `uv` (for hyper-fast dependency resolution) and Python 3.13-slim for ephemeral, secure deployments in any cloud environment.
* **Observability & Tracing:** Integrated directly with LangSmith for real-time latency tracking, token usage monitoring, and node-execution tracing.

## 🚀 Tech Stack
* **Orchestration:** LangGraph / LangChain
* **Vector Database:** Qdrant (Local/Ephemeral)
* **Web Search Tool:** Tavily Search API
* **LLM Engine:** OpenRouter (OpenAI Compatible)
* **UI & Package Management:** Streamlit & `uv`
* **Infrastructure:** Docker & Docker Desktop

## graph TD
    A[User Query] --> B{Router Node}
    B -->|Technical| C[Vector Store Retrieval]
    B -->|Non-Technical| D[Guardrail Refusal]
    C --> E{Grader Node}
    E -->|Relevant| F[Generate Answer]
    E -->|Irrelevant| G[Tavily Web Search]
    G --> F
    F --> H[LangSmith Tracing]

## 🛠 Getting Started

### Prerequisites
* Docker installed and running.
* API Keys for OpenRouter, Tavily, and LangSmith.

### Local Deployment
1. Clone the repository:
   ```bash
   git clone [https://github.com/vishalxai/DocRAGSearch-Modular-Engine.git](https://github.com/vishalxai/DocRAGSearch-Modular-Engine.git)
   cd DocRAGSearch-Modular-Engine

   Create an .env file in the root directory (ensure no quotation marks around values):
   
OPENAI_API_KEY=sk-or-v1-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=DocRAG-Enterprise

**Build and run Docker Container**
docker build -t docrag-engine .
docker run -p 8501:8501 --env-file .env docrag-engine

**🧪 The "3-Strike" Architecture Test
To verify the system's routing logic, self-correction, and enterprise guardrails, upload a technical PDF (e.g., ML Interview Questions) and test these three prompts:

The RAG Test: "What is the bias-variance trade-off?" * Behavior: Routes to Local DB. Extracts and formats the answer strictly from the provided PDF.

The CRAG Fallback Test: "What is the latest version of the LangChain library?" * Behavior: Checks local DB -> Fails Grader Node -> Self-Corrects -> Routes to Web Search -> Delivers real-time internet data.

The Boundary Enforcement Test: "How do I bake a vanilla cake using AI?" * Behavior: LLM identifies the out-of-scope intent and triggers an immediate SECURITY REFUSAL to prevent API waste.


