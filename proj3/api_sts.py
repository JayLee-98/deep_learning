from fastapi import FastAPI, Form

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

app = FastAPI()


@app.post("/sts/")
async def login(sentence1: str = Form(), sentence2: str = Form()):
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    print(embedding1.shape)

    similarities = model.similarity(embedding1, embedding2)
    print(similarities) # 결과값 tensor([[0.5034]]) : 두 문장은 비슷하다고 평가함. 학습시킬때 커뮤니티 글도 모두 학습시키기때문에 어투에서도 유사함을 이해함. 의미론적 유사도

    return {"result": similarities}