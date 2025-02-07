from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
templates=['Trình bày theo phong cách đe dọa người khác ép buộc bên làm (không nên có tên). Giảng giải kĩ các thông tin chi tiết cũng như viết ít nhất 50 dòng cho câu trả lời. Câu trả lời nên là một hoặc nhiều đoạn văn (không nên có gạch đầu dòng). Kết thúc bằng câu đi hoặc sẽ bị đấm.']
class ChatBotAgent():
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
