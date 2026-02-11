# ==========================================
# LEGAL AI – CLOUD SAFE (NO API / NO AUTH)
# ==========================================

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal AI – RAG",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --------------------------------------------------
# UI STYLE (AUTO THEME SAFE)
# --------------------------------------------------
st.markdown("""
<style>
.card {
    background: var(--secondary-background-color);
    padding: 20px;
    border-radius: 14px;
    margin-top: 20px;
}
.answer {
    border-left: 4px solid #10a37f;
    padding: 15px;
    border-radius: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>⚖️ Legal Document Analysis & Q&A</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Cloud-Safe RAG without API Keys</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("## 📂 Upload Legal PDFs")
files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# --------------------------------------------------
# BUILD VECTOR STORE
# --------------------------------------------------
def build_vectorstore(texts, metadata):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks, metas = [], []
    for text, meta in zip(texts, metadata):
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metas.append(meta)

    return FAISS.from_texts(chunks, LocalEmbeddings(), metas)

# --------------------------------------------------
# SIMPLE ANSWER (EXTRACTIVE – SAFE)
# --------------------------------------------------
def answer_question(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    combined = "\n\n".join([doc.page_content for doc in docs])
    return combined[:1200]  # safe output limit

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if files:
    texts, metas = [], []

    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
                metas.append({"source": f.name})

    if st.session_state.vectorstore is None:
        with st.spinner("Indexing documents..."):
            st.session_state.vectorstore = build_vectorstore(texts, metas)

    question = st.text_input("Ask a legal question")

    if question:
        with st.spinner("Searching documents..."):
            answer = answer_question(question, st.session_state.vectorstore)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Question:** {question}")
        st.markdown(f"<div class='answer'>{answer}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📂 Upload legal PDF documents to begin.")
