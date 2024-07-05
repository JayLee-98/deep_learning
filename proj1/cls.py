import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg', 'flag.jpg', 'flag2.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
    # urllib.request.urlretrieve(url, name)


# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img) # test라는 이름으로 resized된 img를 보여줘 라는 의미
#   cv2.waitKey(0) # 0을 누를때까지 기다려라는 의미


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)








# STEP 1: Import the necessary modules. 패키지 가져오는 과정 (추론기)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기 객체 만드는 과정  # 기본 옵션. 모든 모델을 사용하던 해야하는 작업
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite') # 가져올 모델 파일을 우클릭해서 상대경로 복사해서 인자 경로 부분에 붙여넣기
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)


# STEP 3: Load the input image. # 데이터를 가져오는 과정
image = mp.Image.create_from_file(IMAGE_FILENAMES[3]) # 같은 계층에 이미지 파일인 burger.jpg가 있기때문에 이렇게 작성하면 됨.

# STEP 4: Classify the input image. # 추론하고, 결과를 받아오는 과정
classification_result = classifier.classify(image)
# print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it. # 엄밀히 말하면 step 4에서 추론한 값을 가져왔기때문에 사실상 끝인것임. # 사용자에게 보여주기 위한 과정.
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")