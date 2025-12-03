from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import Config

# Initialize Config
config = Config()

# Initialize LLM
llm = ChatOpenAI(
    model=config.LLM_MODEL_NAME,
    temperature=0,
    api_key=config.OPENAI_API_KEY
)

# --- Query Rewriter Chain ---
rephrase_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the question is already standalone, return it as is.
"""

rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ]
)

query_rewriter_chain = rephrase_prompt | llm | StrOutputParser()


# --- RAG Generator Chain ---
rag_system_prompt = """You are an assistant for question-answering tasks. 
Answer the question based ONLY on the following pieces of retrieved context. 
If the answer is not in the context, say that you cannot answer based on the available information.
Do NOT use your internal knowledge.
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        ("human", "{question}"),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()
