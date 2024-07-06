from fastapi import FastAPI, File, UploadFile

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

from PIL import Image

# 매우 중요!! 서버 호출하기 이전에 추론기를 생성해놓아야, 서버는 항상 돌고있기때문에 추론기를 매번 만들 필요없이 재활용할 수 있음.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite') # 가져올 모델 파일을 우클릭해서 상대경로 복사해서 인자 경로 부분에 붙여넣기
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI() 
from PIL import Image
import numpy as np
import io

# 이 부분은 필요없다라고만 알고 넘어가도 됨.
# @app.post("/files/")
# async def create_file(file: bytes = File()):
#     return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile): # 비동기적으로 처리. 헤더(컨텐츠타입, 파일명 등)만 전달함. 실제 파일을 읽어오기위해서는 운영체제로 가져와야함.
    
    content = await file.read() # io 통신에 계속해서 받을 수 있음. 서버에서 이미지를 가져옴.
    # content -> jpg 파일인데.. http 통신에서는 파일이 character type 왔다갔다함.
    # 1. text -> binary : io.BytesIO()
    # 2. binary -> PIL Image : Image.open()

    # STEP 3: Load the input image. # 데이터를 가져오는 과정
    # image = mp.Image.create_from_file(IMAGE_FILENAMES[3]) # create_from_file() 함수는 로컬에 있는 파일을 사용하는 함수임.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. # 추론하고, 결과를 받아오는 과정
    classification_result = classifier.classify(image)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it. # 엄밀히 말하면 step 4에서 추론한 값을 가져왔기때문에 사실상 끝인것임. # 사용자에게 보여주기 위한 과정.
    top_category = classification_result.classifications[0].categories[0]
    result = (f"{top_category.category_name} ({top_category.score:.2f})")
    
    return {"result" : result}

