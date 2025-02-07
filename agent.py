import torch
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
templates=['Trình bày theo phong cách đe dọa người khác ép buộc bên làm (không nên có tên). Giảng giải kĩ các thông tin chi tiết cũng như viết ít nhất 50 dòng cho câu trả lời. Câu trả lời nên là một hoặc nhiều đoạn văn (không nên có gạch đầu dòng). Kết thúc bằng câu đi hoặc sẽ bị đấm.']
snapshot_download(repo_id="capleaf/viXTTS",
                  repo_type="model",
                  local_dir="model")
config = XttsConfig()
config.load_json("./model/config.json")
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
        return self.model.inference(
            text,
            language='vi',
            gpt_cond_latent=gcl,
            speaker_embeddings=se,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )