Hmm retriever is inconsistent
Fix: I use bk ai biencoder to improve retriever
Vector search is not enough, so incorperate BM25
with BM25 I need to perform text processing, 
or I can use semantic chunking to improve the search result.

document_grader = phoranker

result this morning: with phoreranker, I receive a quite decent response, may be I will stop there and start building an interface.
More advanced RAG technique will be learned when my data increases, but for now, it is fine.
Then, I will reorganize my code and push onto github, then building the interface, also note some potential techniques and resources for future use. 

Also read documents from doc data

hmm larges documents size combined make the model suck