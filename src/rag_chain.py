from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# =====================================================
# Embeddings & Vector DB
# =====================================================
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =====================================================
# STRICT RAG PROMPT (SINGLE SOURCE OF TRUTH)
# =====================================================
system_prompt = """
You are a STRICT Retrieval-Augmented Generation (RAG) assistant.

Rules:
- Answer ONLY using information explicitly present in the Context.
- Do NOT use prior knowledge or assumptions.
- Do NOT combine information from different documents unless explicitly stated.
- Short questions (e.g., "address", "phone") refer to facts in Context.
- Greetings should be responded to politely, then answer ONLY if Context allows.

If the answer is NOT clearly stated in Context, reply EXACTLY:
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
    temperature=0.2
)

# =====================================================
# RAG CHAIN
# =====================================================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
