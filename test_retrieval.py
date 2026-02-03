from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# SAME embedding model used to build DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local("knowledge_db", embeddings, allow_dangerous_deserialization=True)

query = "What projects did Charishma build?"
docs = db.similarity_search(query, k=5)

print("FOUND DOCS:", len(docs))
for d in docs:
    print("----")
    print(d.page_content)
