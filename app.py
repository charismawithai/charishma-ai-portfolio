import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- LOAD API ----------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- LOAD VECTOR DB ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("knowledge_db", embeddings, allow_dangerous_deserialization=True)

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions" not in st.session_state:
    st.session_state.questions = []

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
h1, h2, h3, p, div { color: white; }

[data-testid="stSidebar"] {
    padding-top: 10px !important;
}

.chat-box {
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
# ================= SIDEBAR =================
with st.sidebar:

    # Rounded Profile Image (CENTER)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("data/charisma.jpg", width=220)

    st.markdown("""
    <style>
    img {
        border-radius: 100%;
        border: 3px solid #444;
        display:block;
        margin-left:auto;
        margin-right:auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # BIG NAME
    st.markdown("""
    <h2 style="text-align:center; font-weight:800; font-size:22px;">Charishma Devi</h2>
    """, unsafe_allow_html=True)

    # TAGLINE
    st.markdown("""
    <p style="text-align:center; font-size:14px; color:#bbbbbb;">
    Building AI Projects in Public • AI Educator • Tech Career Mentor
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # LinkedIn
    st.markdown("🔗 **LinkedIn**")
    st.markdown("[Visit LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN_USERNAME)")

    # Instagram
    st.markdown("📸 **Instagram**")
    st.markdown("[Visit Instagram](https://instagram.com/YOUR_INSTAGRAM_USERNAME)")

    # Resume Download
    st.markdown("📄 **Resume**")
    with open("data/Charisma_Resume.pdf", "rb") as file:
        st.download_button(
            "📥 Download My Resume",
            file,
            "Charishma_Devi_Resume.pdf",
            mime="application/pdf"
        )

    # Chat History
    st.markdown("---")
    st.subheader("💬 Chat History")
    for q in reversed(st.session_state.questions):
        st.markdown(f"- {q}")

    # Clear Chat Button
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
Answer like a confident AI engineer and educator.

When asked about projects ALWAYS mention:
Career Prediction System,
AI Interview Feedback Analyzer,
Health Risk Prediction System,
Diabetes Risk Prediction System.

Explain professionally like ChatGPT.
Do not invent new projects.
"""

# ---------------- CHAT INPUT ----------------
question = st.chat_input("Ask about my AI journey, projects, or skills...")

# ---------------- AI RESPONSE ----------------
if question:
    # Save question in sidebar
    st.session_state.questions.append(question)

    # Save chat
    st.session_state.messages.append({"role": "user", "content": question})

    # Retrieve RAG context
    docs = db.similarity_search(question, k=10)
    context = " ".join([d.page_content for d in docs])

    # Call AI
    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )

    answer = chat.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
