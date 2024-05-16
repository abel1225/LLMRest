import utils
import json

context_list = []

with open('./dataset/train.json','r', encoding='utf-8') as fp:
    data = json.load(fp)
    for line in data:
        line = (line["context_text"])
        context_list.append(line)

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0")
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0").half().cuda()
prompt_text = "小孩发烧怎么办"
# ChatGLM询问
response, _ = model.chat(tokenizer, prompt_text, history=[])
print(response)
# 文本查询
sim_results= utils.get_top_n_sim_text(query=prompt_text, documents=context_list)
print(sim_results)
# 由ChatGLM根据文本查询输出结果
prompt = utils.generate_prompt(prompt_text, sim_results)
response, _ = model.chat(tokenizer, prompt, history=[])
print(response)
    