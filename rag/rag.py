from langchain_core.messages import HumanMessage, SystemMessage
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from utils import *
import weaviate
import json
from langchain_weaviate.vectorstores import WeaviateVectorStore
import os
from prompt import *
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama

model_name = "keepitreal/vietnamese-sbert"
# VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm = HuggingFaceEndpoint(
    repo_id="vilm/vinallama-7b",
    max_new_tokens=1024,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.3,
    task = "text-generation",
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def retrieve(state):
  
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}
def generate(state):
    # global templat
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([
        # SystemMessage(content=template),
        HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state):

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    # web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        # print("Raw response:", result.content)

        # json_result = json.loads(result.content)
        matches = re.findall(r'\{.*?\}', result, re.DOTALL)  # List of JSON-like strings
        json_result = [json.loads(m) for m in matches] if matches else None  # Convert to JSON
        if json_result and isinstance(json_result, list) and len(json_result) > 0:
            grade = json_result[0].get("binary_score", "no")  # Use .get() to avoid KeyError
        else:
            grade = "no"
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            # web_search = "Yes"
            continue
    return {"documents": filtered_docs}