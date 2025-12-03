from typing import List
from langgraph.graph import StateGraph, END
from src.ai_search.state import AgentState
from src.ai_search.chains import query_rewriter_chain, rag_chain
from src.retrieval.query_engine import QueryEngine
from src.retrieval.similarity_search import SearchResult

def format_docs(docs: List[SearchResult]) -> str:
    """Format retrieved documents for the LLM context."""
    formatted = []
    for doc in docs:
        # SearchResult has 'metadata' attribute of type ChunkMetadata
        source = doc.metadata.filename if doc.metadata and hasattr(doc.metadata, 'filename') else "Unknown source"
        formatted.append(f"Source: {source}\nContent: {doc.text}")
    return "\n\n".join(formatted)

def rewrite_query(state: AgentState):
    """
    Transform the query to produce a better question.
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    messages = state["messages"]
    
    # If there are no messages (first query), we might not need to rewrite, 
    # but it's often good to normalize it anyway.
    # For now, we always rewrite to ensure it's standalone.
    better_question = query_rewriter_chain.invoke({"messages": messages, "question": question})
    return {"question": better_question}

def retrieve(state: AgentState):
    """
    Retrieve documents based on the query.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Initialize QueryEngine
    # In a production app, you might want to inject this dependency 
    # or use a singleton to avoid reloading models.
    engine = QueryEngine()
    
    # Perform search
    results = engine.query(query_text=question, top_k=5)
    
    return {"documents": results}

def generate(state: AgentState):
    """
    Generate answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = format_docs(documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": generation}

def build_graph():
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Define edges
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app
