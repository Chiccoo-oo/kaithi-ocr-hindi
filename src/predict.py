# 


import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from mapping import map_kaithi_to_hindi
from label_to_kaithi import label_to_kaithi

# Load trained model
model = load_model("models/kaithi_ocr.h5")

# Load class indices
with open("models/class_indices.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> label
index_to_label = {v: k for k, v in class_indices.items()}

# Load test image
img = cv2.imread("data/raw/test.png")
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
class_index = int(np.argmax(pred))

predicted_label = index_to_label[class_index]
kaithi_char = label_to_kaithi[predicted_label]
hindi_char = map_kaithi_to_hindi(kaithi_char)

print("Predicted label:", predicted_label)
print("Kaithi:", kaithi_char)
print("Hindi:", hindi_char)
