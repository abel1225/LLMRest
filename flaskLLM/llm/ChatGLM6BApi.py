from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0")
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0").half().cuda()
model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
