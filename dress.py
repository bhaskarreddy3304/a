# ==========================================
# LEGAL AI – CLOUD SAFE (FREE DEPLOYMENT)
# ==========================================

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.llms import HuggingFaceHub

from sentence_transformers import SentenceTransformer

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
if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------------------------
# THEME (AUTO – STREAMLIT SAFE)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}
.card {
    background: var(--secondary-background-color);
    border-radius: 14px;
    padding: 20px;
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
    "<h1 style='text-align:center;'>⚖️ Legal Document Analysis & Q&A (RAG)</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Cloud-Safe Legal AI using Retrieval-Augmented Generation</p>",
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
### ✅ Enabled Features
• Legal PDF analysis  
• Document-based Q&A  
• RAG with FAISS  
• Free-tier safe deployment  

### ❌ Removed for Cloud Safety
• Voice input/output  
• OS-level access  
• Threads & background tasks  
""")

# --------------------------------------------------
# EMBEDDINGS (CLOUD SAFE)
# --------------------------------------------------
class HFEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# --------------------------------------------------
# BUILD RAG (HUGGINGFACE HUB – SAFE)
# --------------------------------------------------
@st.cache_resource
def build_rag(chunks, meta):
    vector_db = FAISS.from_texts(chunks, HFEmbeddings(), meta)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.3, "max_length": 256}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever()
    )

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if files:
    texts, metas = [], []

    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                texts.append(content)
                metas.append({"source": f.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks, meta = [], []
    for t, m in zip(texts, metas):
        for c in splitter.split_text(t):
            chunks.append(c)
            meta.append(m)

    qa = build_rag(chunks, meta)

    question = st.text_input("Ask a legal question based on the uploaded documents")

    if question:
        with st.spinner("Analyzing legal documents…"):
            result = qa(question)["result"]

        st.session_state.chat.append((question, result))

    if st.session_state.chat:
        q, a = st.session_state.chat[-1]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Question:** {q}")
        st.markdown(f"<div class='answer'>{a}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📂 Upload legal PDF documents to start analysis.")
