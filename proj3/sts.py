from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

sentence1 = "네가 일라오이만 하면 무조건 승리하더라?"
sentence2 = "내가 아는 사람들 중에 네가 게임 제일 잘함."

# 2. Calculate embeddings by calling model.encode()
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
print(embedding1.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embedding1, embedding2)
print(similarities) # 결과값 tensor([[0.5034]]) : 두 문장은 비슷하다고 평가함. 학습시킬때 커뮤니티 글도 모두 학습시키기때문에 어투에서도 유사함을 이해함. 의미론적 유사도
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])