from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


# =====================================================
# IN-MEMORY CHAT STORE (SAFE)
# =====================================================
store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Returns chat history for a given session.
    History is used ONLY for conversational flow,
    NOT for factual answering.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# =====================================================
# MEMORY WRAPPER (DOES NOT CHANGE FACTS)
# =====================================================
def with_memory(chain):
    """
    Wraps an existing RAG chain with chat memory.
    Memory does NOT influence factual answers.
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
