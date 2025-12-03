from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from src.retrieval.similarity_search import SearchResult

class AgentState(TypedDict):
    """
    State for the RAG agent graph.
    """
    messages: List[BaseMessage]
    question: str
    documents: List[SearchResult]
    generation: str
