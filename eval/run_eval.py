import os
import sys
import json
import warnings
from dotenv import load_dotenv
from datasets import Dataset

# Suppress annoying Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate
# FIX 1: We only import faithfulness to completely bypass the OpenRouter Embedding 401 error
from ragas.metrics import faithfulness 

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore.qdrant_manager import QdrantManager
from src.graph_builder.builder import RAGGraphBuilder
from langchain_openai import ChatOpenAI

os.environ["USER_AGENT"] = "DocRAG-Eval-Suite/1.0"

def execute_benchmarks():
    load_dotenv()
    
    db_manager = QdrantManager()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    retriever = db_manager.get_retriever()
    builder = RAGGraphBuilder(retriever, llm)
    rag_app = builder.build()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "eval_dataset.json"), "r") as f:
        gold_data = json.load(f)
    
    questions, answers, contexts, ground_truths = [], [], [], []
    
    print("🚀 Invoking LangGraph for Evaluation Queries...")
    for item in gold_data:
        q = item["question"]
        print(f"\nEvaluating: {q}")
        
        # Invoke Graph for the Answer
        response = rag_app.invoke({"question": q, "steps": []})
        
        # FIX 2: Invoke Retriever directly to guarantee we capture the text for Ragas
        retrieved_docs = retriever.invoke(q)
        
        questions.append(q)
        answers.append(response.get("answer", response.get("generation", "")))
        ground_truths.append(item["ground_truth"])
        
        # Extract the text and append it to our contexts list
        doc_texts = [doc.page_content for doc in retrieved_docs]
        contexts.append(doc_texts)

    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print("\n📊 Calculating Metrics with Ragas Judge...")
    result = evaluate(
        ds, 
        metrics=[faithfulness], # Only evaluating Faithfulness
        llm=eval_llm
    )
    
    print("\n✅ Benchmark Results:")
    print(result.to_pandas())

if __name__ == "__main__":
    execute_benchmarks()