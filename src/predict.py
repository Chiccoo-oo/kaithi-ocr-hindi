import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.mapping import kaithi_to_hindi

model = load_model("../models/kaithi_ocr.h5")

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (64,64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
label_index = pred.argmax()

kaithi_char = list(kaithi_to_hindi.keys())[label_index]
print("Hindi Output:", kaithi_to_hindi[kaithi_char])
