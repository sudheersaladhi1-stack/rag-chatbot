from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

from dotenv import load_dotenv

load_dotenv()

# =====================================================
# LLM
# =====================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# =====================================================
# STRICT QA PROMPT
# =====================================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a retrieval-augmented assistant.

You MUST answer using ONLY the provided context.

Allowed:
- You MAY rephrase or summarize information explicitly present in the context.
- You MAY answer definition-style questions (e.g., "What is X?")
  if the context clearly describes X, even if not in dictionary format.
- You MAY combine information from a SINGLE retrieved chunk.

Not allowed:
- Do NOT use external knowledge.
- Do NOT answer if the information is not clearly supported by the context.
- Do NOT combine information across unrelated documents.

If the answer cannot be reasonably derived from the context, reply EXACTLY:
"I don't know based on the provided context."

Context:
{context}
"""
        ),
        ("human", "{input}")
    ]
)

# =====================================================
# BASE RAG CHAIN (NO RETRIEVAL HERE)
# =====================================================
_base_chain = (
    qa_prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# CHAT MEMORY STORE
# =====================================================
store = {}

def _get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =====================================================
# ✅ EXPORTED RAG CHAIN WITH MEMORY
# =====================================================
rag_chain_with_memory = RunnableWithMessageHistory(
    _base_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
