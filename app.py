import streamlit as st
from uuid import uuid4
import os, re, hashlib, requests, html
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

from src.rag_chain import rag_chain
from src.rag_chat_memory import rag_chain_with_memory, store


# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("PDF / TXT / URL → Strict RAG (No Hallucination)")

# =====================================================
# Embeddings & Vectorstore
# =====================================================
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

CHROMA_DIR = "chroma_db"

def get_vectorstore(collection: str):
    return Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

def get_retriever(collection: str):
    return get_vectorstore(collection).as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

# =====================================================
# Utilities
# =====================================================
def normalize_query(q: str):
    q = q.strip()
    if not q:
        return None
    if len(q) < 3:
        return None
    if not re.search(r"[a-zA-Z]", q):
        return None
    return q

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def extract_person_names(text: str):
    return {w.lower() for w in re.findall(r"[A-Z][a-z]+", text)}

def highlight_text(text: str, query: str):
    text = html.escape(text)
    words = re.findall(r"\w+", query.lower())

    for word in set(words):
        if len(word) < 3:
            continue
        pattern = re.compile(rf"({re.escape(word)})", re.IGNORECASE)
        text = pattern.sub(
            r"<mark style='background-color:#ffe066'>\1</mark>",
            text,
        )
    return text

# =====================================================
# URL Loader
# =====================================================
def load_url_as_documents(url: str):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()

    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": urlparse(url).netloc,
                "type": "url",
                "url": url,
            },
        )
    ]

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", "default")

st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "PDF / TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

st.sidebar.header("🌐 Add Website URL")
url_input = st.sidebar.text_input("Enter website URL")

# =====================================================
# Ingestion
# =====================================================
def ingest_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore(collection_name)

    for i, c in enumerate(chunks):
        content = c.page_content.strip()
        if not content or len(content) < 30:
            continue

        # 🔐 ultra-safe metadata
        metadata = {
            "source": str(c.metadata.get("source", "unknown")),
            "collection": str(collection_name),
            "chunk": str(i),
        }

        chunk_id = hashlib.md5(
            f"{collection_name}:{uuid4().hex}:{i}".encode()
        ).hexdigest()

        try:
            vs.add_texts(
                texts=[content],
                metadatas=[metadata],
                ids=[chunk_id],
            )
        except Exception as e:
            st.warning(f"Skipped one chunk due to ingestion error")
            continue


# =====================================================
# Ingest Actions
# =====================================================
if st.sidebar.button("📥 Ingest documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:
        docs = []
        for f in uploaded_files:
            tmp = f"tmp_{f.name}"
            with open(tmp, "wb") as t:
                t.write(f.read())

            loader = (
                PyPDFLoader(tmp)
                if f.name.endswith(".pdf")
                else TextLoader(tmp)
            )
            docs.extend(loader.load())
            os.remove(tmp)

        ingest_documents(docs)
        st.sidebar.success("Documents ingested ✅")
        st.rerun()

if st.sidebar.button("🌍 Ingest URL"):
    if not url_input:
        st.sidebar.warning("Enter a valid URL")
    else:
        ingest_documents(load_url_as_documents(url_input))
        st.sidebar.success("Website ingested ✅")
        st.rerun()

# =====================================================
# Clear Knowledge Base
# =====================================================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)

    store.clear()
    st.session_state.clear()
    st.sidebar.success("Knowledge base cleared ✅")
    st.rerun()

# =====================================================
# Session State
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])

# =====================================================
# Disable chat if empty
# =====================================================
doc_count = get_vectorstore(collection_name)._collection.count()
st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("Upload documents or a URL to start.")
    st.stop()

# =====================================================
# Show chat history
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# Chat
# =====================================================
user_input = st.chat_input("Ask a question based on the uploaded knowledge")

if user_input:
    normalized_query = normalize_query(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔹 Greeting / weak query handling
    if normalized_query is None:
        answer = "Hello 👋 How can I help you?"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.stop()

    # 🔹 Retrieval (crash-proof)
    raw_docs = []
    try:
        retriever = get_retriever(collection_name)
        raw_docs = retriever.invoke(normalized_query)
    except Exception:
        try:
            data = get_vectorstore(collection_name)._collection.get(
                include=["documents", "metadatas"],
                limit=3,
            )
            raw_docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(
                    data.get("documents", []),
                    data.get("metadatas", []),
                )
            ]
        except Exception:
            raw_docs = []

    # 🔍 Debug panel
    with st.expander("🔍 Retrieved chunks (highlighted)"):
        st.write(f"Retrieved {len(raw_docs)} chunks")
        for i, d in enumerate(raw_docs[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(
                highlight_text(d.page_content[:1000], normalized_query),
                unsafe_allow_html=True,
            )

    # 🔹 Deduplicate
    seen, docs = set(), []
    for d in raw_docs:
        t = d.page_content.strip()
        if t and t not in seen:
            seen.add(t)
            docs.append(d)
        if len(docs) == 3:
            break

    # 🔹 Strict answering
    if not docs:
        answer = "I don't know based on the provided context."
    else:
        context = format_docs(docs)
        if extract_person_names(normalized_query) - extract_person_names(context):
            answer = "I don't know based on the provided context."
        else:
            answer = rag_chain_with_memory.invoke(
                {"input": normalized_query, "context": context},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
