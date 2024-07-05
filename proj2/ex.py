# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 추론기 생성
app = FaceAnalysis()
# app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3 데이터 불러오기
# from insightface.data import get_image as ins_get_image
# img = ins_get_image('t1')

img1 = cv2.imread("iu1.jpg") # 인터넷에서 찾은 아이유 사진으로 변경
img2 = cv2.imread("iu2.jpg")

# STEP 4 추론
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# print(faces[0]) # 이렇게 데이터를 출력할 수 있음 # 맨 처음
# print(faces1[0]) # 아이유 얼굴 임베딩 확인용

# STEP 5 추론한 값 가공
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./iu_output.jpg", rimg)

# then print all-to-all face similarity
# embedding의 값에 -4, -8, 5, 1 이렇게 있으면 연산이 힘들어서 normalization해줘야함. 그걸 normed_embedding
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32) # 그런데 normed_embedding은 순수 파이썬이기 때문에 np로 행렬로 바꿔줘야함.
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sim = np.dot(feat1, feat2.T) # np.dot이라는 임베딩 행렬의 곱은 코사인 유사도를 나타냄.
print(sim) # 결과로 나오는 소수점은 퍼센트가 아님. -1 ~ 1 사이의 결과값이 나옴. 동일인물 여부를 말하지 않음. 