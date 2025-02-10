import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vietnam_number import n2w
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langgraph.graph import StateGraph
from langgraph.graph import START, StateGraph
from langgraph.graph import END
from graph_state.graph_state import GraphState
from rag.rag import *
templates=['Trình bày theo phong cách đe dọa người khác ép buộc bên làm (không nên có tên). Giảng giải kĩ các thông tin chi tiết cũng như viết ít nhất 50 dòng cho câu trả lời. Câu trả lời nên là một hoặc nhiều đoạn văn (không nên có gạch đầu dòng). Kết thúc bằng câu đi hoặc sẽ bị đấm.']
snapshot_download(repo_id="capleaf/viXTTS",
                  repo_type="model",
                  local_dir="model")
config = XttsConfig()
config.load_json("./model/config.json")
class ChatBotAgent():
    def __init__(self, template_type, graph_state="langgraph"):
        self.llm = HuggingFaceEndpoint(
            repo_id="vilm/vinallama-7b",
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
            temperature=0.3,
            task="text-generation",
        )
        self.template=templates[template_type]
        self.gs = StateGraph(GraphState)
        if graph_state == "langgraph":
            self.gs = StateGraph(GraphState)
            self.gs.add_node("retrieve", retrieve)
            self.gs.add_node("grade_documents", grade_documents)
            self.gs.add_node("generate", generate)
            # self.gs.add_node("grade_generation_v_documents_and_question", grade_generation_v_documents_and_question)
            self.gs.add_node(
                "inform_no_information",
                lambda: "Xin lỗi, nhưng chúng tôi không được cung cấp thông tin cho vấn đề này.",
                # end=True,  # Marks this as the final step
            )

            self.gs.add_edge("retrieve", "grade_documents")
            self.gs.add_edge("grade_documents", "generate")
            self.gs.add_conditional_edges(
                "generate",
                grade_generation_v_documents_and_question,
                {
                    "not supported": "generate",
                    "useful": END,
                    "not useful": "inform_no_information",
                    "max retries": END,
                },
            )
            self.gs.set_entry_point("retrieve")
    def answer(self, db, query):
        app = self.gs.compile()

class TTSAgent():
    def __init__(self):
        self.model = Xtts.init_from_config(config).load_checkpoint(config, checkpoint_dir='./model/').eval()
    def get_gcl_se(self):
        return self.model.get_conditioning_latents(
            audio_path="./model/vi_sample.wav",
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )
    def extract_audio(self, text, gcl, se):
        words = text.split(" ")
        new_text = ""
        for word in words:
            if word.isdigit():
                new_text += n2w(word)
            new_text += word
        audio = self.model.inference(
            new_text,
            language='vi',
            gpt_cond_latent=gcl,
            speaker_embeddings=se,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )
        return audio["wav"]