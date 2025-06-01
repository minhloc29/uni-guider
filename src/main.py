from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets `src/`
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Move to `uni_guider/
sys.path.append(os.path.abspath(BASE_DIR))

from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uuid import uuid4
from chatbot.rag.prompt import template
from chatbot.rag.config import LLM_CONFIG, EMBEDDING_CONFIG
import chatbot.rag.retrieval as retrieval
from pydantic import BaseModel
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from chatbot.utils import *
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZSjeZMOA6igO9MiVLBXoFEdjBiyNYiSA"

data = load_doc_from_json(filename="data/chunking_documents.json")

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_CONFIG["model_name"],
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
index = faiss.IndexFlatL2(len(hf.embed_query("Xin chào.")))
vectorstore = FAISS(
    embedding_function=hf,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(data))]
vectorstore.add_documents(documents=data, ids=uuids)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins. You can specify a list of domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")),
    name="static",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("intro_page.html", {"request": request})


@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

import asyncio

async def stream_response(query, reference_urls):
    print("Handle stream response")
    loop = asyncio.get_running_loop()
    for chunk in await loop.run_in_executor(None, lambda: llm.stream(query)):   
        text = chunk.content
        replaced_text = text.replace("*", "")
        yield replaced_text

    yield "\n\nCâu trả lời được trích từ nguồn:\n\n" 
    for url in reference_urls:
        yield url + "\n"  
    yield "\n"  
    
@app.post("/ask")
async def get_question_stream(data: QuestionRequest):
    question = data.question
    print(f"Question: {question}")
    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    print("Received question:", question)

    # Retrieve relevant documents
    retrieved_docs = retrieval.retrieve_documents(question, retriever)
    reference_urls = []
    for doc in retrieved_docs:
        reference_urls.append(doc.metadata['source'])  
    doc_txts = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Prepare the LLM prompt
    query = [
        SystemMessage(f"{template}\nNgữ cảnh: \n{doc_txts}"),
        HumanMessage(question),
    ]

    # Return streaming response
    return StreamingResponse(stream_response(query, reference_urls), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8060)

# Học bổng khuyến khích học tập bao nhiêu tiền
#nêu điều kiện nhận học bổng toshiba