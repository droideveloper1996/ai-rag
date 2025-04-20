import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException,UploadFile, HTTPException,File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import hashlib
TRACK_FILE = "processed_files.json"
if os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = []

load_dotenv()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates/rag_agent")

# Optional: CORS support for external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")

# Pinecone Init
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "multi-pdf-rag"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

pdf_folder = "./static/knowledge/"  # Your folder with PDFs
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
 
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

 
for pdf_file in pdf_files:
    if pdf_file in processed_files:
        print(f"✅ Skipping already processed file: {pdf_file}")
        continue

    path = os.path.join(pdf_folder, pdf_file)
    text = extract_text_from_pdf(path)
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks)

    for i, chunk in enumerate(chunks):
        vector_id = f"{pdf_file}_chunk_{i}"
        index.upsert([(vector_id, embeddings[i].tolist(), {"source": pdf_file, "text": chunk})])

    processed_files.append(pdf_file)
    print(f"📄 Processed and indexed: {pdf_file}")

with open(TRACK_FILE, "w") as f:
    json.dump(processed_files, f, indent=2)

print("✅ All PDFs processed and stored in Pinecone.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/query")
async def query_handler(request: Request):
    try:
        body = await request.json()
        user_query = body.get("query", "").strip()

        if not user_query:
            raise HTTPException(status_code=400, detail="Query is empty.")

        # Step 1: Embed query
        query_embedding = embedding_model.encode(user_query).tolist()

        # Step 2: Retrieve top chunks from Pinecone
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        matches = results.get("matches", [])

        if not matches:
            return JSONResponse({"answer": "Sorry, I couldn't find relevant information."})

        # Step 3: Extract context
        retrieved_chunks = [match["metadata"]["text"] for match in matches if "text" in match["metadata"]]
        context = "\n\n".join(retrieved_chunks)

        # Step 4: Generate prompt and answer
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {user_query}
Answer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()
        return JSONResponse({"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_hash = hashlib.md5(file.filename.encode()).hexdigest()
    if file_hash in processed_files:
        return {"message": "File already processed."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to extract text: {str(e)}"})
    finally:
        os.remove(tmp_path)

    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks)

    for i, chunk in enumerate(chunks):
        vector_id = f"{file_hash}_chunk_{i}"
        index.upsert([(vector_id, embeddings[i].tolist(), {"source": file.filename, "text": chunk})])

    processed_files.append(file_hash)
    with open(TRACK_FILE, "w") as f:
        json.dump(processed_files, f, indent=2)

    return {"message": f"✅ Uploaded and indexed {file.filename}", "chunks": len(chunks)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)