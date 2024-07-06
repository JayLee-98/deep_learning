from transformers import pipeline

# question_answerer = pipeline("question-answering", model="my_awesome_qa_model") # model에 제작자 이름이 안들어가있기때문에 수동으로 stevhliu/ 추가해줘야함.
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

question = "How many programming languages does SiRi support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages, otherwise SiRi has 500 billion parameters" # score 확률값, start와 end는 그 단서를 찾은 곳

result = question_answerer(question=question, context=context)

print(result) # 질문을 물어보면 context를 기반으로 대답을 해줌. 