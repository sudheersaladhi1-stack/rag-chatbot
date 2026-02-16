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
            """
You MUST answer strictly and only from the provided Context.

Special rule for greetings:
- If the user's input is ONLY a greeting (e.g., hi, hello, hey),
  respond with a polite greeting.
- After greeting, answer ONLY if Context contains relevant information.
- Otherwise say exactly:
"I don't know based on the provided context."

Rules:
- Use ONLY information explicitly present in Context.
- DO NOT use chat history as knowledge.
- DO NOT infer or guess.
- If the answer is NOT explicitly stated in Context, reply EXACTLY:
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
