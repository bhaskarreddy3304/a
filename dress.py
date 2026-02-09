import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal AI – Voice Enabled",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------------------------
# UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
html, body, .stApp { background-color:#0b0f19; color:#e5e7eb; }
h1, h2 { color:#e5e7eb; }
input, textarea {
    background:#020617!important;
    color:#e5e7eb!important;
    border:1px solid #6366f1!important;
}
.user { background:#1e293b; padding:12px; border-radius:10px; margin:8px 0; }
.bot  { background:#020617; padding:12px; border-radius:10px; border-left:4px solid #6366f1; }
.source { font-size:13px; color:#94a3b8; }
.btn {
    background:#6366f1; color:white;
    padding:8px 14px; border-radius:8px;
    border:none; cursor:pointer;
}
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>⚖️ Legal AI – Speech Enabled Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI Answers • Voice Input • Streamlit Cloud Safe</p>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

embedder = load_embedder()
llm = load_llm()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📂 Upload Legal PDFs")
files = st.sidebar.file_uploader("PDF only", type=["pdf"], accept_multiple_files=True)

# --------------------------------------------------
# TEXT SPLITTER
# --------------------------------------------------
def split_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_index(files):
    texts, sources = [], []

    for f in files:
        reader = PdfReader(f)
        full_text = ""
        for p in reader.pages:
            if p.extract_text():
                full_text += p.extract_text()

        chunks = split_text(full_text)
        texts.extend(chunks)
        sources.extend([f.name] * len(chunks))

    embeddings = embedder.encode(texts)
    nn = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn.fit(embeddings)

    return nn, texts, sources

# --------------------------------------------------
# BROWSER SPEECH + TTS
# --------------------------------------------------
st.markdown("""
<script>
function startDictation() {
    if (!('webkitSpeechRecognition' in window)) {
        alert("Speech recognition not supported");
        return;
    }
    const r = new webkitSpeechRecognition();
    r.lang = "en-IN";
    r.onresult = e => {
        document.getElementById("speech_input").value = e.results[0][0].transcript;
    };
    r.start();
}
function speakText(text) {
    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = "en-IN";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
}
</script>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if files:
    nn, texts, sources = build_index(files)

    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input("Ask a legal question", key="speech_input")
    with col2:
        st.markdown('<button class="btn" onclick="startDictation()">🎤 Speak</button>', unsafe_allow_html=True)

    if question:
        with st.spinner("Analyzing legal documents..."):
            q_emb = embedder.encode([question])
            _, idxs = nn.kneighbors(q_emb)

            context = " ".join([texts[i] for i in idxs[0]])
            prompt = f"Answer the legal question clearly.\n\nContext:\n{context}\n\nQuestion:\n{question}"

            answer = llm(prompt)[0]["generated_text"]

        st.session_state.chat.append((question, answer, idxs[0]))

    # CHAT DISPLAY
    for q, a, idxs in st.session_state.chat[::-1]:
        st.markdown(f"<div class='user'><b>Q:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot'>{a}</div>", unsafe_allow_html=True)

        srcs = set([sources[i] for i in idxs])
        for s in srcs:
            st.markdown(f"<div class='source'>Source: {s}</div>", unsafe_allow_html=True)

        st.markdown(
            f'<button class="btn" onclick="speakText(`{a[:2000]}`)">🔊 Read Answer</button>',
            unsafe_allow_html=True
        )
else:
    st.info("⬅️ Upload legal PDF documents to begin.")
