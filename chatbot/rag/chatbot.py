from langchain_core.messages import HumanMessage, SystemMessage
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from utils import *
import weaviate
import json
from langchain_weaviate.vectorstores import WeaviateVectorStore
from chatbot.rag.config import Rag, RagConfig
from langchain_huggingface import HuggingFaceEndpoint

model_name = "keepitreal/vietnamese-sbert"
# VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# vilm/vinallama-7b
# VietAI/vit5-base
# Thay tên model huggingface vô cái repo_id
llm = HuggingFaceEndpoint(
    repo_id="vilm/vinallama-7b",
    max_new_tokens=1024,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.3,
    task = "text-generation",
)

data = load_doc_from_json(filename="data/document_langchain.json")
weaviate_client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore.from_documents(data, hf, client=weaviate_client)
retriever = vectorstore.as_retriever(k = 5)
#docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.28.4

config = RagConfig(retriever = retriever, llm = llm)
app = Rag(config)
app = app.compile()

question = input("Bạn muốn hỏi gì: ")
inputs = {
    "question": question,
    "max_retries": 3,
}
for event in app.stream(inputs, stream_mode="values"):
    print(event)