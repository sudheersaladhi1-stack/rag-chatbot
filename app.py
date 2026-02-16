"""Streamlit RAG Chatbot — PDF / TXT / URL ingestion with strict RAG answering."""

import hashlib
import html
import logging
import os
import re
import tempfile
from uuid import uuid4

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse

from src.rag_chat_memory import rag_chain_with_memory, store

# Load .env once, at the entry point
load_dotenv()

logger = logging.getLogger(__name__)

# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("PDF / TXT / URL → Strict RAG (No Hallucination)")

CHROMA_DIR = "chroma_db"


# =====================================================
# Cached resources  (FIX: was re-instantiated on every rerun)
# =====================================================
@st.cache_resource
def get_embedding_model() -> SentenceTransformerEmbeddings:
    """Load the sentence-transformer embedding model once per process."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_vectorstore(collection: str) -> Chroma:
    """Return a cached Chroma vectorstore for the given collection."""
    return Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_model(),
    )


def get_retriever(collection: str):
    return get_vectorstore(collection).as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )


# =====================================================
# Utilities
# =====================================================
def normalize_query(q: str) -> str | None:
    """Return a cleaned query string, or None if the query is too weak."""
    q = q.strip()
    if not q:
        return None
    if len(q) < 3:
        return None
    if not re.search(r"[a-zA-Z]", q):
        return None
    return q


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def highlight_text(text: str, query: str) -> str:
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
def load_url_as_documents(url: str) -> list[Document]:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

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
# Ingestion
# FIX: collection_name is now an explicit parameter,
#      not captured from outer scope via closure.
# =====================================================
def ingest_documents(docs: list[Document], collection: str) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    vs = get_vectorstore(collection)

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if not content or len(content) < 30:
            continue

        metadata = {
            "source": str(chunk.metadata.get("source", "unknown")),
            "collection": str(collection),
            "chunk": str(i),
        }

        chunk_id = hashlib.md5(
            f"{collection}:{uuid4().hex}:{i}".encode()
        ).hexdigest()

        try:
            vs.add_texts(
                texts=[content],
                metadatas=[metadata],
                ids=[chunk_id],
            )
        except Exception as exc:  # FIX: include exception detail in the warning
            st.warning(f"Skipped chunk {i}: {exc}")
            logger.warning("Chunk %d ingestion failed: %s", i, exc)


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
# Ingest Actions
# =====================================================
if st.sidebar.button("📥 Ingest documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:
        docs: list[Document] = []
        for f in uploaded_files:
            # FIX: use tempfile to avoid name collisions and CWD pollution
            suffix = ".pdf" if f.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                docs.extend(loader.load())
            finally:
                os.remove(tmp_path)

        ingest_documents(docs, collection_name)
        st.sidebar.success("Documents ingested ✅")
        st.rerun()

if st.sidebar.button("🌍 Ingest URL"):
    if not url_input:
        st.sidebar.warning("Enter a valid URL")
    else:
        ingest_documents(load_url_as_documents(url_input), collection_name)
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
# Disable chat if knowledge base is empty
# =====================================================
doc_count = get_vectorstore(collection_name)._collection.count()
st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("Upload documents or a URL to start.")
    st.stop()

# =====================================================
# Chat history
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

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Greeting / weak query — respond and stop
    if normalized_query is None:
        answer = "Hello 👋 How can I help you?"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()

    # ── Retrieval (crash-proof) ───────────────────────────────────────────────
    raw_docs: list[Document] = []
    try:
        retriever = get_retriever(collection_name)
        raw_docs = retriever.invoke(normalized_query)
    except Exception as exc:  # FIX: log instead of silently swallowing
        logger.warning("MMR retrieval failed (%s), falling back to raw fetch.", exc)
        st.warning(f"Retrieval degraded, using fallback. ({exc})")
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
        except Exception as exc2:
            logger.error("Fallback retrieval also failed: %s", exc2)
            st.error(f"Could not retrieve any documents: {exc2}")
            raw_docs = []

    # ── Debug panel ──────────────────────────────────────────────────────────
    with st.expander("🔍 Retrieved chunks (highlighted)"):
        st.write(f"Retrieved {len(raw_docs)} chunks")
        for i, d in enumerate(raw_docs[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(
                highlight_text(d.page_content[:1000], normalized_query),
                unsafe_allow_html=True,
            )

    # ── Deduplicate ──────────────────────────────────────────────────────────
    seen: set[str] = set()
    docs: list[Document] = []
    for d in raw_docs:
        t = d.page_content.strip()
        if t and t not in seen:
            seen.add(t)
            docs.append(d)
        if len(docs) == 3:
            break

    # ── Answer ───────────────────────────────────────────────────────────────
    if not docs:
        answer = "I don't know based on the provided context."
    else:
        context = format_docs(docs)
        # FIX: removed unreliable extract_person_names() guard — it matched any
        # capitalised word (months, cities, common nouns) causing false refusals.
        # The LLM's own strict system prompt handles hallucination prevention.
        answer = rag_chain_with_memory.invoke(
            {"input": normalized_query, "context": context},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})