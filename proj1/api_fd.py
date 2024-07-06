from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='models/fd/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

app = FastAPI()

from PIL import Image
import io

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()

    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)
    print(detection_result)
    counts = len(detection_result.detections)
    object_list = []

#   DetectionResult(
#     detections=[Detection(bounding_box=BoundingBox(origin_x=488, origin_y=177, width=184, height=184),
#                            categories=[Category(index=0, score=0.7779484987258911, display_name=None, category_name=None)], 
#                                 , 
#                 Detection(bounding_box=BoundingBox(origin_x=799, origin_y=152, width=173, height=173), 
#                            categories=[Category(index=0, score=0.6154318451881409, display_name=None, category_name=None)], 
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  NormalizedKeypoint(x=0.7395403385162354, 
#                                                                                 y=0.21685484051704407, label='', score=0.0)])])

    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category) 
    print(object_list)

    # STEP 5: Process the detection result. In this case, visualize it.

    # cv2_imshow(rgb_annotated_image)
    # cv2.imshow("test", rgb_annotated_image)
    # cv2.waitKey(0)

    if counts is None:
        return{
            "result": "침입자 없음",
            "counts": counts,

        }
    else:
        return{
            "result": "침입자 있음",
            "counts": counts,
        }