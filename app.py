import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- LOAD ENV ----------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Charishma Devi — AI Portfolio",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
h1, h2, h3, p, div { color: white; }
[data-testid="stSidebar"] { padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions" not in st.session_state:
    st.session_state.questions = []

# ---------------- LOAD EMBEDDINGS (CPU SAFE) ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # 🔥 CRITICAL FIX
    )

embeddings = load_embeddings()

# ---------------- LOAD VECTOR DB SAFELY ----------------
db = None
if os.path.exists("knowledge_db"):
    try:
        db = FAISS.load_local(
            "knowledge_db",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.warning("⚠️ Knowledge base not loaded. Using fallback answers.")

# ================= SIDEBAR =================
with st.sidebar:

    st.image("charisma.jpg", width=260)
    st.markdown("<h2 style='text-align:center;'>Charishma Devi</h2>", unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; color:#bbbbbb;">
    Building AI Projects in Public • AI Educator • Tech Career Mentor
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("🔗 **LinkedIn**")
    st.markdown("[Visit LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN_USERNAME)")

    st.markdown("📸 **Instagram**")
    st.markdown("[Visit Instagram](https://instagram.com/YOUR_INSTAGRAM_USERNAME)")

    st.markdown("📄 **Resume**")
    with open("Charisma_Resume.pdf", "rb") as file:
        st.download_button(
            "📥 Download My Resume",
            file,
            "Charishma_Devi_Resume.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    st.subheader("💬 Chat History")
    for q in reversed(st.session_state.questions):
        st.markdown(f"- {q}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.questions = []
        st.rerun()

# ================= MAIN UI =================
st.title("🤖 Charishma Devi — Personal AI Portfolio Assistant")
st.write("Ask anything about my AI projects, skills, LinkedIn journey, or career goals.")

# ---------------- SYSTEM PROMPT ----------------
system_prompt = """
You are Charishma Devi speaking in FIRST PERSON.

STRICT RULES (NO EXCEPTIONS):
- Answer ONLY using the provided context.
- If the answer is not present in the context, say:
  "This information is not available in my knowledge base yet."
- NEVER guess colleges, cities, universities, or timelines.
- NEVER use outside knowledge.

Education facts MUST come only from context.

Projects that MUST be mentioned when asked:
- Career Prediction System
- AI Interview Feedback Analyzer
- Health Risk Prediction System
- Diabetes Risk Prediction System


Do not invent projects.
"""

# ---------------- CHAT INPUT ----------------
question = st.chat_input("Ask about my AI journey, projects, or skills...")

# ---------------- RESPONSE ----------------
if question:
    st.session_state.questions.append(question)
    st.session_state.messages.append({"role": "user", "content": question})

    context = ""
    if db:
        docs = db.similarity_search(question, k=6)
        context = " ".join(d.page_content for d in docs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

