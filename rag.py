import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# -------------------------------
# 🔐 OpenAI Setup
# -------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 📄 PDF Extraction
# -------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# -------------------------------
# 🔗 ChromaDB Setup
# -------------------------------
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",  # for persistence
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name="multi_doc_rag")

# -------------------------------
# 🔠 Embedding + Chunking Setup
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 📥 Step 1: Process Multiple PDFs
# -------------------------------
pdf_dir = "./documents"  # 📁 Folder with your PDFs
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

doc_counter = 0

for pdf_file in pdf_files:
    file_path = os.path.join(pdf_dir, pdf_file)
    text = extract_text_from_pdf(file_path)
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"{pdf_file}_chunk_{i}"],
            metadatas=[{"source": pdf_file}]
        )
    doc_counter += 1

print(f"✅ {doc_counter} PDFs processed and indexed into ChromaDB.")

# -------------------------------
# ❓ Step 2: Accept Query and Search
# -------------------------------
user_query = input("Enter your query: ")
query_embedding = embedding_model.encode(user_query).tolist()

results = collection.query(query_embeddings=[query_embedding], n_results=5)
retrieved_chunks = results["documents"][0]
sources = results["metadatas"][0]

# -------------------------------
# 🧠 Step 3: OpenAI Prompting
# -------------------------------
context = "\n\n".join(retrieved_chunks)
source_list = "\n".join([f"- {meta['source']}" for meta in sources])

prompt = f"""
You are a helpful AI assistant. Use the following context from documents to answer the user's question.

Context:
{context}

Sources:
{source_list}

Question: {user_query}
Answer:
"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content
print("\nAnswer:\n", answer)