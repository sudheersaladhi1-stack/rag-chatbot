from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter



# =====================================================
# STRICT SYSTEM PROMPT (NO HALLUCINATION)
# =====================================================
system_prompt = """
You are a retrieval-augmented assistant.

You MUST answer strictly and only from the provided Context.

Special rule for greetings:
- If the user's input is ONLY a greeting (hi, hello, hey),
  you may respond politely.
- Do NOT add any factual information unless it exists in Context.

Rules:
- Use ONLY information explicitly present in Context.
- DO NOT use prior knowledge.
- DO NOT infer or guess.
- DO NOT explain anything not stated in Context.
- If the answer cannot be found in Context, reply EXACTLY:
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
    model="gpt-4o-mini",   # best cost / quality for RAG
    temperature=0.2,
)

# =====================================================
# RAG CHAIN (NO RETRIEVAL INSIDE)
# Context MUST be passed from app.py
# =====================================================
rag_chain = (
    {
        # Get 'context' from the dictionary, otherwise pass it through
        "context": itemgetter("context"), 
        # Get 'input' from the dictionary
        "input": itemgetter("input")
    }
    | prompt
    | llm
    | StrOutputParser()
)
# =====================================================
# Local test (optional)
# =====================================================
if __name__ == "__main__":
    test_context = "Machine learning is a subset of artificial intelligence."
    question = "What is machine learning?"
    answer = rag_chain.invoke(
        {"input": question, "context": test_context}
    )
    print("Q:", question)
    print("A:", answer)
