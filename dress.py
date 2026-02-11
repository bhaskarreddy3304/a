# ==========================================
# LEGAL DOCUMENT ANALYSIS & Q&A USING RAG
# STREAMLIT CLOUD – STABLE VERSION
# ==========================================

import streamlit as st
import io

from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG (FAST LOAD)
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
# UI STYLES (THEME SAFE)
# --------------------------------------------------
st.markdown("""
<style>
.card {
    background-color: var(--secondary-background-color);
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
}
.answer {
    background-color: rgba(16,163,127,0.12);
    border-left: 4px solid #10a37f;
    padding: 16px;
    border-radius: 10px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>⚖️ Legal Document Analysis & Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>RAG-based Legal AI (Streamlit Cloud Safe)</p>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📂 Upload Legal PDFs")
files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 Features
✅ Multi-PDF Analysis  
✅ Retrieval-Augmented Generation  
✅ Legal Question Answering  
✅ PDF Answer Download  
✅ Cloud Stable Deployment  
""")

# --------------------------------------------------
# EMBEDDINGS CLASS
# --------------------------------------------------
class HFEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# --------------------------------------------------
# LOAD LLM (LAZY + CACHED)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# BUILD RAG PIPELINE (LAZY)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_rag(chunks, meta):
    vectorstore = FAISS.from_texts(
        chunks,
        HFEmbeddings(),
        meta
    )

    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------
def generate_pdf(question, answer):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("<b>LEGAL AI – GENERATED REPORT</b>", styles["Title"]))
    story.append(Paragraph("<br/><b>Question:</b>", styles["Heading2"]))
    story.append(Paragraph(question, styles["Normal"]))
    story.append(Paragraph("<br/><b>Answer:</b>", styles["Heading2"]))
    story.append(Paragraph(answer, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
if files:
    texts, metas = [], []

    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
                metas.append({"source": file.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks, meta = [], []
    for t, m in zip(texts, metas):
        for chunk in splitter.split_text(t):
            chunks.append(chunk)
            meta.append(m)

    qa = build_rag(chunks, meta)

    question = st.text_input("Ask a legal question")

    if question:
        with st.spinner("Analyzing legal documents..."):
            answer = qa(question)["result"]

        st.session_state.chat.append((question, answer))

    if st.session_state.chat:
        q, a = st.session_state.chat[-1]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Question:** {q}")
        st.markdown(f"<div class='answer'>{a}</div>", unsafe_allow_html=True)

        st.download_button(
            "📄 Download Answer as PDF",
            data=generate_pdf(q, a),
            file_name="Legal_AI_Report.pdf",
            mime="application/pdf"
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📂 Upload legal PDF documents to start analysis.")
