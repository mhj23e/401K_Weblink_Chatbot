import io
import os
import re
from typing import Dict, List

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from pypdf import PdfReader

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ------------------------------
# 1. GLOBALS & INITIAL SETUP
# ------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = np.array([])
if "model" not in st.session_state:
    st.session_state.model = None

repo_id = "tiiuae/falcon-7b-instruct"


def clear_session_data() -> None:
    """Hard reset in-memory user/session data."""
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = np.array([])
    st.session_state.model = None


def load_llm():
    """Load Hugging Face Hub model lazily."""
    return HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.3, "max_new_tokens": 256},
    )


@st.cache_resource
def get_embedding_model():
    """Load sentence-transformer once per process, with safe fallback."""
    if os.getenv("FORCE_SIMPLE_EMBEDDINGS") == "1" or SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


# ------------------------------
# 2. INGESTION HELPERS
# ------------------------------
def fetch_website_text(url: str) -> str:
    """Fetch and parse text from a URL."""
    if not re.match(r"^https?://", url):
        st.error("URL must start with http:// or https://")
        return ""

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    except Exception as exc:
        st.error(f"Failed to fetch URL content: {exc}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    text_parts: List[str] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text based on file extension."""
    suffix = uploaded_file.name.lower().split(".")[-1]
    file_bytes = uploaded_file.getvalue()

    if suffix in {"txt", "md", "csv"}:
        return file_bytes.decode("utf-8", errors="ignore")
    if suffix == "pdf":
        try:
            return extract_text_from_pdf(file_bytes)
        except Exception as exc:
            st.warning(f"Could not read PDF '{uploaded_file.name}': {exc}")
            return ""

    st.warning(f"Unsupported file type skipped: {uploaded_file.name}")
    return ""


def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """Split source documents into retrieval chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_records: List[Dict[str, str]] = []

    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunk_records.append({"source": doc["source"], "chunk_id": str(idx), "text": piece})

    return chunk_records


def _simple_embed(texts: List[str], dim: int = 512) -> np.ndarray:
    """Fast local embedding fallback (hashing trick) for demo reliability."""
    mat = np.zeros((len(texts), dim), dtype=float)
    for i, text in enumerate(texts):
        for token in re.findall(r"\w+", text.lower()):
            mat[i, hash(token) % dim] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def embed_chunks(chunks: List[Dict[str, str]]) -> np.ndarray:
    """Generate normalized embeddings for chunks."""
    texts = [chunk["text"] for chunk in chunks]
    model = get_embedding_model()

    if model is None:
        st.info("Using local fallback embeddings for retrieval.")
        return _simple_embed(texts)

    vectors = np.array(model.encode(texts))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def retrieve_relevant_chunks(query: str, chunks: List[Dict[str, str]], chunk_embeddings: np.ndarray, top_k: int = 3) -> List[Dict[str, str]]:
    """Return top-k chunk records by cosine similarity."""
    model = get_embedding_model()
    if model is None:
        query_vector = _simple_embed([query])[0]
    else:
        query_vector = np.array(model.encode([query])[0])
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)

    similarities = np.dot(chunk_embeddings, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def answer_query_with_context(query: str, relevant_chunks: List[Dict[str, str]]) -> str:
    """Generate answer grounded in retrieved chunks with fallback behavior."""
    if st.session_state.model is None:
        st.session_state.model = load_llm()

    context = "\n\n".join(f"[Source: {chunk['source']} | Chunk: {chunk['chunk_id']}]\n{chunk['text']}" for chunk in relevant_chunks)
    prompt = f"""You are a retrieval assistant.
Answer only using the context below. If context is insufficient, say you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        return st.session_state.model(prompt)
    except Exception as exc:
        st.warning(f"Model generation failed, returning context-based fallback answer. Details: {exc}")
        fallback = "\n\n".join(chunk["text"][:300] for chunk in relevant_chunks)
        return f"I could not run the generation model. Here is the most relevant source context:\n\n{fallback}"


# ------------------------------
# 3. STREAMLIT UI
# ------------------------------
st.title("401k Chatbot")
st.caption(
    "Load data from a URL and/or uploaded documents (PDF/TXT/MD/CSV), then ask questions. "
    "Your data stays in-memory only for this session and is removed when the session ends or when you clear it."
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Clear session data now"):
        clear_session_data()
        st.success("Session data cleared.")
with col_b:
    st.info("No user data is written to disk by this app.")

url = st.text_input("Optional: Paste a website URL (must include http/https)")
uploaded_files = st.file_uploader(
    "Optional: Upload one or more files",
    type=["pdf", "txt", "md", "csv"],
    accept_multiple_files=True,
)

if st.button("Load Data Sources"):
    clear_session_data()
    docs: List[Dict[str, str]] = []

    if url.strip():
        url_text = fetch_website_text(url.strip())
        if url_text:
            docs.append({"source": f"URL: {url.strip()}", "text": url_text})

    for uploaded_file in uploaded_files or []:
        extracted = extract_text_from_uploaded_file(uploaded_file)
        if extracted.strip():
            docs.append({"source": f"FILE: {uploaded_file.name}", "text": extracted})

    if not docs:
        st.warning("No usable content found. Add a URL, upload files, or both.")
    else:
        st.session_state.chunks = chunk_documents(docs)
        st.session_state.chunk_embeddings = embed_chunks(st.session_state.chunks)
        st.success(f"Loaded {len(docs)} source(s) and {len(st.session_state.chunks)} chunk(s).")

question = st.text_input("Ask a question about the loaded content")

if st.button("Get Answer"):
    if not st.session_state.chunks:
        st.warning("Please load data sources first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        relevant = retrieve_relevant_chunks(question, st.session_state.chunks, st.session_state.chunk_embeddings)
        answer = answer_query_with_context(question, relevant)

        st.text_area("Answer", answer, height=180)
        st.markdown("### Retrieved Sources")
        for idx, chunk in enumerate(relevant, start=1):
            st.markdown(f"**{idx}. {chunk['source']} (chunk {chunk['chunk_id']})**")
            st.caption(chunk["text"][:350] + ("..." if len(chunk["text"]) > 350 else ""))
