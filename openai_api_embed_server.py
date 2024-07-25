"""
# * @Author: DuTim
# * @Date: 2024-07-25 10:39:06
# * @LastEditTime: 2024-07-25 11:29:54
# * @Description: 本地模型mock openai 风格 embedding 服务
"""
import argparse
from pprint import pprint
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

app = FastAPI()


encode_model = None
tokenizer = None


class Item(BaseModel):
    input:  Union[str, list[str]]
    model: str
    prefix: str = "为这个句子生成表示以用于检索相关文章："
    encoding_format: str = None


# beg_prefix = "为这个句子生成表示以用于检索相关文章："


@app.post("/v1/embeddings")
async def create_embedding(item: Item):
    # 确保输入是字符串列表
    texts = [str(item.prefix + x) for x in item.input]

    # 计算token数量
    tokens = tokenizer(texts, padding=True, truncation=True)
    token_count = sum(len(ids) for ids in tokens["input_ids"])

    # 生成嵌入
    with torch.no_grad():
        embeddings = encode_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    # 将张量转换为列表
    embeddings_list = embeddings.tolist()

    # 构建响应
    data = [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings_list)]

    return {
        "object": "list",
        "data": data,
        "model": item.model,
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
    }  # 改回原来的模型名称


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地模型mock openai 风格 embedding 服务")
    parser.add_argument("--port", type=int, default=17302, help="server port")
    parser.add_argument("--model_name_or_path", default="/media/nfs2/xz_yyb/xz/download/bge-large-zh-v1.5", type=str, help="model_name_or_path")
    parser.add_argument("--gpu", type=str, default="0", help="device")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载模型和分词器
    encode_model = SentenceTransformer(args.model_name_or_path, device="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 打印解析后的参数
    pprint(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
