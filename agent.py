from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
class Agent():
    def __init__(self, template_type):
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-pro',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.template=templates[template_type]
    def answer(self, db, query):
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=db.as_retriever()
        )
        response = chain.invoke(
            {"question": query + self.template},
            return_only_outputs=True,
        )
        return response['answer']