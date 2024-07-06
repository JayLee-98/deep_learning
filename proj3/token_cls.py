# STEP 1
from transformers import pipeline

# STEP 2 추론기 생성 pipeline에 어떤 모델을 쓸 건지와 모델 경로를 명시함
classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

# STEP 3 데이터
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
# text = "My favorite coffee shop is within 5 minutes walking distance."
# text = "Korean government is suffering from the lowest polling rates."

# STEP 4 추론
result = classifier(text)

# STEP 5 추론값 활용
print(result)