from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# =====================================================
# System Prompt (STRICT)
# =====================================================
system_prompt = """You MUST answer strictly and only from the provided Context.

Rules:
- Use ONLY information explicitly present in Context.
- DO NOT use prior knowledge.
- DO NOT guess.
- DO NOT infer missing information.
- If the answer is NOT explicitly stated in Context, reply EXACTLY:
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
