# STEP 1
from transformers import pipeline

# STEP 2 추론기 생성 pipeline에 어떤 모델을 쓸 건지와 모델 경로를 명시함
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model") # stevhliu가 만든 모델이다. 깃허브에서 / 앞이 닉네임과 같은 개념. 스페이스 라고 부름.

# STEP 3 데이터
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다."

# STEP 4 추론
result = classifier(text)

# STEP 5 추론값 활용
print(result)