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
def grade_generation_v_documents_and_question(state):

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation
    )
    result = llm.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    # print("Raw response:", result.content)

    # json_result = json.loads(result.content)
    matches = re.findall(r'\{.*?\}', result, re.DOTALL)  # List of JSON-like strings
    json_result = [json.loads(m) for m in matches] if matches else None  # Convert to JSON
    if json_result and isinstance(json_result, list) and len(json_result) > 0:
        grade = json_result[0].get("binary_score", "no")  # Use .get() to avoid KeyError
    else:
        grade = "no"
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation
        )
        result = llm.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )

        # json_result = json.loads(result.content)
        matches = re.findall(r'\{.*?\}', result, re.DOTALL)  # List of JSON-like strings
        json_result = [json.loads(m) for m in matches] if matches else None  # Convert to JSON
        if json_result and isinstance(json_result, list) and len(json_result) > 0:
            grade = json_result[0].get("binary_score", "no")  # Use .get() to avoid KeyError
        else:
            grade = "no"
        # if "binary_score" in json_result:
        #     grade = json_result["binary_score"]
        # else:
        #     print("Warning: 'binary_score' key not found in JSON result")
        #     grade = 'no'

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"
