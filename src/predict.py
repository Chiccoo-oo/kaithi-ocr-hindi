import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mapping import map_kaithi_to_hindi
from label_to_kaithi import label_to_kaithi


model = load_model("models/kaithi_ocr.h5")

img = cv2.imread("data/raw/test.jpeg")
img = cv2.resize(img, (64,64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
class_index = np.argmax(pred)

class_labels = list(label_to_kaithi.keys())
predicted_label = class_labels[class_index]

kaithi_char = label_to_kaithi[predicted_label]
hindi_char = map_kaithi_to_hindi(kaithi_char)

print("Predicted label:", predicted_label)
print("Kaithi:", kaithi_char)
print("Hindi:", hindi_char)
