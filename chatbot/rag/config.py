import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import re
import json
from chatbot.rag.prompt import *
@dataclass
class GraphState:
    question: str  # User question
    generation: str = None  # LLM generation
    web_search: str = None  # Binary decision to run web search
    max_retries: int = 3  # Max number of retries for answer generation
    answers: int = 0  # Number of answers generated
    loop_step: int = 0
    documents: List[str] = field(default_factory=list)  # List of retrieved documents
    
@dataclass
class RagConfig:
    retriever: any
    llm: any
    rag_prompt: str = rag_prompt
    doc_grader_prompt: str = doc_grader_prompt
    doc_grader_instructions: str = doc_grader_instructions
    hallucination_grader_instructions: str = hallucination_grader_instructions
    hallucination_grader_prompt: str = hallucination_grader_prompt
    answer_grader_prompt: str = answer_grader_prompt
    answer_grader_instructions: str = answer_grader_prompt

class Rag:
    def __init__(self, config: RagConfig):
        self.config = config
        self.workflow = StateGraph(GraphState)
        self._build_workflow()

    def _build_workflow(self):
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node(
            "inform_no_information",
            lambda state: "Xin lỗi, nhưng chúng tôi không được cung cấp thông tin cho vấn đề này.",
        )

        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_edge("grade_documents", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "inform_no_information",
                "max retries": END,
            },
        )
        self.workflow.set_entry_point("retrieve")

    def grade_generation_v_documents_and_question(self, state: GraphState):

        print("---CHECK HALLUCINATIONS---")
        question = state.question
        documents = state.documents
        generation = state.generation
        max_retries = state.max_retries  # Default to 3 if not provided
        hallucination_grader_prompt_formatted = self.config.hallucination_grader_prompt.format(
            documents=self.format_docs(documents), generation=generation
        )
        result = self.config.llm.invoke(
            [SystemMessage(content=self.config.hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
       
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
            answer_grader_prompt_formatted = self.config.answer_grader_prompt.format(
                question=question, generation=generation
            )
            result = self.config.llm.invoke(
                [SystemMessage(content=self.config.answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )

            # json_result = json.loads(result.content)
            matches = re.findall(r'\{.*?\}', result, re.DOTALL)  # List of JSON-like strings
            json_result = [json.loads(m) for m in matches] if matches else None  # Convert to JSON
            if json_result and isinstance(json_result, list) and len(json_result) > 0:
                grade = json_result[0].get("binary_score", "no")  # Use .get() to avoid KeyError
            else:
                grade = "no"
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful" 

            elif state.loop_step <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"  

            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"  

        elif state.loop_step <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"  

        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
        
    def compile(self):
        return self.workflow.compile()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve(self, state: GraphState):
        print("---RETRIEVE---")
        question = state.question
        documents = self.config.retriever.invoke(question)
        state.documents = documents
        return state

    def generate(self, state):
        print("---GENERATE---")
        question = state.question
        documents = state.documents
        loop_step = state.loop_step

        docs_txt = self.format_docs(documents)
        rag_prompt_formatted = self.config.rag_prompt.format(context=docs_txt, question=question)
        generation = self.config.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        state.generation = generation
        state.loop_step = loop_step + 1
        return state

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state.question
        documents = state.documents

        filtered_docs = []
        for d in documents:
            doc_grader_prompt_formatted = self.config.doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.config.llm.invoke(
                [SystemMessage(content=self.config.doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )

            matches = re.findall(r'\{.*?\}', result, re.DOTALL)
            json_result = [json.loads(m) for m in matches] if matches else None
            if json_result and isinstance(json_result, list) and len(json_result) > 0:
                grade = json_result[0].get("binary_score", "no")
            else:
                grade = "no"

            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        state.documents = filtered_docs
        return state