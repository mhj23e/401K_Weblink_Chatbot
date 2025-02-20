import streamlit as st
import requests
from bs4 import BeautifulSoup

# LangChain and retrieval
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# HuggingFace LLM
from langchain.llms import HuggingFaceHub

# ------------------------------
# 1. GLOBALS & INITIAL SETUP
# ------------------------------
# We'll store our chunks and embeddings in Streamlit's session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = []
    st.session_state.model = None

# Using a public instruct model from HF. 
# If you're using a private or bigger model, set the HUGGINGFACEHUB_API_TOKEN env variable.
repo_id = "tiiuae/falcon-7b-instruct"  # or "gpt2" or "bigscience/bloom-560m"

# Instantiate an LLM from Hugging Face Hub
# If you have a HF API token, it will pick it up from your environment
# or you can pass `huggingfacehub_api_token="YOUR_TOKEN"` as an argument.
def load_llm():
    return HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.7, "max_length": 256}
    )

# Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------
def fetch_website_text(url: str) -> str:
    """Fetch and parse text from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = "\n".join(p.get_text() for p in paragraphs if p.get_text())
            return text
        else:
            st.error(f"Request returned status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return ""

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> list:
    """Split the text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

def embed_chunks(chunks: list) -> np.ndarray:
    """Generate embeddings for each text chunk."""
    return embedding_model.encode(chunks)

def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray, top_k=3) -> list:
    """Return the most relevant chunks based on dot-product similarity."""
    query_embedding = embedding_model.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding)
    # Sort by similarity desc
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def answer_query_with_context(query: str, relevant_chunks: list):
    """Send the relevant context to the LLM for a final answer, without repeating the context."""
    if st.session_state.model is None:
        st.session_state.model = load_llm()
    
    # Combine relevant chunks into a single prompt
    context = "\n".join(relevant_chunks)
    prompt = f"""Use the following context to answer the question:

Context:
{context}
        
Question:
{query}

Answer:
"""
    response = st.session_state.model(prompt)
    return response

# ------------------------------
# 3. STREAMLIT UI
# ------------------------------
st.title("401k Chatbot")

st.write("""
Paste a URL, let the app fetch & embed its text, then ask questions about it. 
All data is stored **ephemerally** in session state, so nothing persists after a session.
""")

# URL input
url = st.text_input("Paste your website link here (must include http/https):")

if st.button("Load Website"):
    # Clear session data each time we load a new website
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = []
    
    raw_text = fetch_website_text(url)
    if raw_text:
        # Chunk & embed
        st.session_state.chunks = chunk_text(raw_text)
        st.session_state.chunk_embeddings = embed_chunks(st.session_state.chunks)
        st.success("Website data loaded! Now you can ask questions.")

# Q&A section
question = st.text_input("Ask a question about the website content:")

if st.button("Get Answer"):
    if not st.session_state.chunks:
        st.warning("Please load a website first!")
    else:
        # Retrieve relevant chunks
        relevant = retrieve_relevant_chunks(
            question,
            st.session_state.chunks,
            st.session_state.chunk_embeddings
        )
        # Generate final answer
        answer = answer_query_with_context(question, relevant)
        st.text_area("Answer", answer, height=150)
