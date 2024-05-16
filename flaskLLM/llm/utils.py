import BM25Okapi from bm25

# 文本相关性(相似度)的比较算法BM25(还可使用余弦相关性计算)
# query 是需要查询的文本, documents 为文本库, top_n 为返回最接近的n条文本内容
def get_top_n_sim_text(query: str, documents:List[str], top_n=3):
    tokenized_corpus=[]
    for doc in documents:
        text = []
        for char in doc:
            text.append(char)
        tokenized_corpus.append(text)
    
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = [char for char in query]
    # doc_scores = bm25.get_scores(tokenized_query) # array([0., 0.93729472, 0.])

    results = bm25.get_top_n(tokenized_query, tokenized_corpus, top_n)
    results = ["".join(res) for res in results]

    return results

def generate_prompt(question: str, relevant_chunks: List[str]):
    prompt = f'根据文档内容来回答问题,问题是"{question}", 文档内容如下'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt

from transformers import BertTokenizer, BertModel
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # first element of model_output contains all token embeddings 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)