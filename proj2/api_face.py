from fastapi import FastAPI, File, UploadFile


# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


# STEP 2 추론기 생성
face = FaceAnalysis()
# app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))
face.prepare(ctx_id=0, det_size=(640,640))

target_face = []

app = FastAPI()

# @app.post("/registFace/")
# async def create_upload_file(file: UploadFile):

#     content = await file.read()

#     # STEP 3 데이터 불러오기
#     # img1 = cv2.imread("iu1.jpg") # 인터넷에서 찾은 아이유 사진으로 변경
#     nparr = np.fromstring(content, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # STEP 4 추론
#     faces1 = face.get(img)
#     assert len(faces1)==1

#     # STEP 5 추론한 값 가공
#     target_face = np.array(faces1[0].normed_embedding, dtype=np.float32) 
#     print(target_face)
#     return{"result": len(faces1)} # 임베딩 값을 내보낼 수 있음. 이렇게 .normed_embedding하지 않으면 에러발생함.

@app.post("/registFace/")
async def registFace(file: UploadFile):
    content = await file.read()
    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1
    # STEP 5
    target_face.append(np.array(faces1[0].normed_embedding, dtype=np.float32))
    print(target_face)
    return {"result":len(faces1)}

# @app.post("/compareFace/")
# async def create_upload_file(file: UploadFile):

#     content = await file.read()

#     # STEP 3 데이터 불러오기
#     # img1 = cv2.imread("iu1.jpg") # 인터넷에서 찾은 아이유 사진으로 변경
#     nparr = np.fromstring(content, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # STEP 4 추론
#     faces1 = face.get(img)
#     assert len(faces1)==1

#     # STEP 5 추론한 값 가공
#     test_face = np.array(faces1[0].normed_embedding, dtype=np.float32) 
#     sim = np.dot(target_face, test_face.T)
#     return{"result": sim.item()}
@app.post("/compareFace/")
async def compareFace(file: UploadFile):
    content = await file.read()
    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1
    # STEP 5
    test_face = np.array(faces1[0].normed_embedding, dtype=np.float32)
    sim = np.dot(target_face[0], test_face.T)
    return {"result":sim.item()}