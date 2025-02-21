from sentence_transformers import CrossEncoder
from pyvi.ViTokenizer import tokenize
from typing import List
from langchain_core.documents import Document
import numpy as np

MODEL_ID = 'itdainb/PhoRanker'
MAX_LENGTH = 512

model = CrossEncoder(MODEL_ID, max_length=MAX_LENGTH)
model.model.half()

def retrieve_documents(query: str, retriever) -> List[Document]:
    docs = retriever.invoke(query)

    segmented_question = tokenize(query)
    segmented_documents = [tokenize(doc.page_content) for doc in docs]
    tokenized_pairs = [[segmented_question, sent] for sent in segmented_documents]

    scores = model.predict(tokenized_pairs)
    top_ids = np.argsort(scores)[::-1][:2]
    top_documents = [docs[i] for i in top_ids]
    return top_documents
