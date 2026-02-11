# ==========================================
# LEGAL AI – FULL FEATURE SAFE DEPLOYMENT
# ==========================================

import streamlit as st
import io
import os
import time
import threading

# ---------- SAFE VOICE IMPORT ----------
VOICE_AVAILABLE = True
try:
    import speech_recognition as sr
    import pyttsx3
except:
    VOICE_AVAILABLE = False

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

if "speaking" not in st.session_state:
    st.session_state.speaking = False

# --------------------------------------------------
# UI STYLE (AUTO THEME SAFE)
# --------------------------------------------------
st.markdown("""
<style>
.card {
    background-color: var(--secondary-background-color);
    border-radius: 16px;
    padding: 20px;
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
st.markdown("<h1 style='text-align:center;'>⚖️ Legal Document Analysis & Q&A using RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Voice-Enabled AI Assistant (Cloud Safe)</p>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📂 Upload Legal PDFs")
files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎙 Voice Status")
st.sidebar.success("Enabled" if VOICE_AVAILABLE else "Disabled (Cloud Mode)")

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
class HFEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# --------------------------------------------------
# BUILD RAG
# --------------------------------------------------
@st.cache_resource
def build_rag(chunks, meta):
    db = FAISS.from_texts(chunks, HFEmbeddings(), meta)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300
    )

    return RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=pipe),
        retriever=db.as_retriever()
    )

# --------------------------------------------------
# VOICE FUNCTIONS (SAFE)
# --------------------------------------------------
def voice_input():
    if not VOICE_AVAILABLE:
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

def speak(text):
    if not VOICE_AVAILABLE:
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------
def generate_pdf(q, a):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>LEGAL AI REPORT</b>", styles["Title"]),
        Paragraph("<br/><b>Question:</b>", styles["Heading2"]),
        Paragraph(q, styles["Normal"]),
        Paragraph("<br/><b>Answer:</b>", styles["Heading2"]),
        Paragraph(a, styles["Normal"]),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if files:
    texts, metas = [], []

    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
                metas.append({"source": f.name})

    splitter = RecursiveCharacterTextSplitter(800, 150)

    chunks, meta = [], []
    for t, m in zip(texts, metas):
        for c in splitter.split_text(t):
            chunks.append(c)
            meta.append(m)

    qa = build_rag(chunks, meta)

    col1, col2 = st.columns([6, 1])

    with col1:
        question = st.text_input("Ask a legal question")

    with col2:
        if st.button("🎤") and VOICE_AVAILABLE:
            spoken = voice_input()
            if spoken:
                question = spoken

    if question:
        with st.spinner("Analyzing documents..."):
            answer = qa(question)["result"]
        st.session_state.chat.append((question, answer))

    if st.session_state.chat:
        q, a = st.session_state.chat[-1]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Question:** {q}")
        st.markdown(f"<div class='answer'>{a}</div>", unsafe_allow_html=True)

        if st.button("🔊 Read Answer") and VOICE_AVAILABLE:
            threading.Thread(target=speak, args=(a,), daemon=True).start()

        st.download_button(
            "📄 Download PDF",
            generate_pdf(q, a),
            "Legal_AI_Report.pdf",
            "application/pdf"
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload legal documents to begin.")
