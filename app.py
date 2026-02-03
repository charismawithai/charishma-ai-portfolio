import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- LOAD ENV ----------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LOAD PRE-BUILT RAG ----------------
DB_PATH = "knowledge_db"

if not os.path.exists(DB_PATH):
    st.error("Knowledge database not found. Please upload knowledge_db folder.")
    st.stop()

db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- UI ----------------
st.title("🤖 Charishma Devi — Personal AI Portfolio Assistant")

SYSTEM_PROMPT = """
You are Charishma Devi speaking in FIRST PERSON.
Answer ONLY using the retrieved context.
If not found, say you haven't mentioned it yet.
"""

question = st.chat_input("Ask about my AI journey, projects, or skills")

if question:
    docs = db.similarity_search(question, k=6)
    context = " ".join(d.page_content for d in docs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    st.write(response.choices[0].message.content)
