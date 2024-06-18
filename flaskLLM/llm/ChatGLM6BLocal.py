import os
import torch
from transformers import pipeline, AutoTokenizer,AutoConfig,AutoModel

LOCAL_MODEL_DIR = "/llm/chatglm/chatglm3-6b"

MODEL_PATH = os.environ.get("MODEL_PATH", LOCAL_MODEL_DIR)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

if 'cuda' in device:
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().cuda()
else:
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(device)

model = model.eval()

def doChatGlm(message: str):

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    # 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
    response, history = model.chat(tokenizer, message, history=history)
    print(response)
    return response
