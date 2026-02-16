from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# =====================================================
# System Prompt (STRICT)
# =====================================================
system_prompt = """You are a retrieval-augmented assistant.

You MUST answer the question using ONLY the provided context.
Do NOT use prior knowledge.
Do NOT guess.
Do NOT explain concepts that are not explicitly present in the context.

If the answer cannot be found in the context, reply exactly with:
"I don't know based on the provided context."

Context:
{context}
"""



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# =====================================================
# LLM
# =====================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)


# =====================================================
# RAG CHAIN (NO RETRIEVAL HERE)
# =====================================================
rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)
