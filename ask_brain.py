from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embeddings and DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("knowledge_db", embeddings, allow_dangerous_deserialization=True)

question = input("Ask your AI Brain: ")

docs = db.similarity_search(question, k=3)
context = " ".join([d.page_content for d in docs])

system_prompt = """
You are Charishma's personal AI assistant.
Use the retrieved context as factual grounding.
Explain professionally in detailed natural language like ChatGPT.
Do not invent facts beyond the context.
"""

chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=600,
    messages=[
        {"role":"system","content": system_prompt},
        {"role":"user","content": f"Context: {context}\nQuestion: {question}"}
    ]
)

print("\n🧠 AI Answer:\n")
print(chat.choices[0].message.content)
