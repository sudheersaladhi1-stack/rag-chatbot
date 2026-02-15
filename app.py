import streamlit as st
from uuid import uuid4
import os, re, hashlib, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

from src.rag_chat_memory import rag_chain_with_memory, store


# =====================================================
# Utilities
# =====================================================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def extract_named_entities(text: str):
    return {
        w.lower()
        for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    }


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
# Streamlit Config
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("PDF / TXT / URL → Strict RAG (No Hallucination)")


# =====================================================
# Vector Store
# =====================================================
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def get_vectorstore(collection):
    return Chroma(
        collection_name=collection,
        persist_directory="chroma_db",
        embedding_function=embedding_model,
    )


@st.cache_resource(show_spinner=False)
def load_retriever(collection):
    return get_vectorstore(collection).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.4,  # 🔒 STRICT GATE
        },
    )


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", "default")

st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

st.sidebar.header("🌐 Add Website URL")
url_input = st.sidebar.text_input("Enter website URL")

retriever = load_retriever(collection_name)


# =====================================================
# Ingest Helper
# =====================================================
def ingest_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    def make_id(text, src):
        return hashlib.md5(
            f"{collection_name}:{src}:{text}".encode()
        ).hexdigest()

    unique = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        c.metadata["collection"] = collection_name
        uid = make_id(c.page_content, src)
        unique[uid] = c

    vs = get_vectorstore(collection_name)
    vs.add_documents(list(unique.values()), ids=list(unique.keys()))


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

            loader = PyPDFLoader(tmp) if f.name.endswith(".pdf") else TextLoader(tmp)
            docs.extend(loader.load())
            os.remove(tmp)

        ingest_documents(docs)
        st.cache_resource.clear()
        st.sidebar.success("Documents added ✅")
        st.rerun()


if st.sidebar.button("🌍 Ingest URL"):
    if not url_input:
        st.sidebar.warning("Enter a valid URL")
    else:
        ingest_documents(load_url_as_documents(url_input))
        st.cache_resource.clear()
        st.sidebar.success("Website added ✅")
        st.rerun()


# =====================================================
# Clear KB
# =====================================================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)

    store.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.sidebar.success("Knowledge base cleared ✅")
    st.rerun()


# =====================================================
# Session State
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])


# =====================================================
# Disable Chat if Empty
# =====================================================
doc_count = get_vectorstore(collection_name)._collection.count()
st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("Upload documents or a URL to start.")
    st.stop()


# =====================================================
# Display History
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# Chat (STRICT RAG)
# =====================================================
user_input = st.chat_input("Ask a question based on the uploaded knowledge")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    raw_docs = retriever.invoke(user_input)

    if not raw_docs:
        answer = "I don't know based on the provided context."
    else:
        docs, seen = [], set()
        for d in raw_docs:
            text = d.page_content.strip()
            if text and text not in seen:
                seen.add(text)
                docs.append(d)
            if len(docs) == 3:
                break

        if not docs:
            answer = "I don't know based on the provided context."
        else:
            context = format_docs(docs)

            if extract_named_entities(user_input) - extract_named_entities(context):
                answer = "I don't know based on the provided context."
            else:
                answer = rag_chain_with_memory.invoke(
                    {"input": user_input, "context": context},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    },
                )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
