import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ================= LOAD ENV =================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="Charishma Devi — AI Portfolio Assistant",
    layout="wide"
)

# ================= EMBEDDINGS =================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ================= BUILD / LOAD VECTOR DB =================
DB_PATH = "knowledge_db"

@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_PATH):
        loader = TextLoader("my_notes.txt", encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        return db
    else:
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

db = load_vector_db()

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions" not in st.session_state:
    st.session_state.questions = []

# ================= STYLING =================
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
h1,h2,h3,p,div { color:white; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:

    st.image("charisma.jpg", width=260)

    st.markdown("<h2 style='text-align:center;'>Charishma Devi</h2>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#bbbbbb;'>"
        "Building AI Projects in Public • AI Educator • Tech Career Mentor"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("🔗 **LinkedIn**")
    st.markdown("[Visit LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN_USERNAME)")

    st.markdown("📸 **Instagram**")
    st.markdown("[Visit Instagram](https://instagram.com/YOUR_INSTAGRAM_USERNAME)")

    st.markdown("📄 **Resume**")
    with open("Charisma_Resume.pdf", "rb") as f:
        st.download_button(
            "📥 Download My Resume",
            f,
            file_name="Charishma_Devi_Resume.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    st.subheader("💬 Chat History")

    for q in reversed(st.session_state.questions):
        st.markdown(f"- {q}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.questions = []
        st.rerun()

# ================= MAIN =================
st.title("🤖 Charishma Devi — Personal AI Portfolio Assistant")
st.write("Ask anything about my AI projects, skills, LinkedIn journey, or career goals.")

SYSTEM_PROMPT = """
You are Charishma Devi speaking in FIRST PERSON.

You must answer ONLY using the provided context.
If something is not in the context, say:
"I haven't mentioned that information yet."

You MUST ALWAYS mention these projects when relevant:
Career Prediction System
AI Interview Feedback Analyzer
Health Risk Prediction System
Diabetes Risk Prediction System

Never invent colleges, locations, or degrees.
"""

question = st.chat_input("Ask about my AI journey, projects, or skills...")

if question:
    st.session_state.questions.append(question)
    st.session_state.messages.append({"role": "user", "content": question})

    docs = db.similarity_search(question, k=6)
    context = " ".join(d.page_content for d in docs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ================= CHAT DISPLAY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
