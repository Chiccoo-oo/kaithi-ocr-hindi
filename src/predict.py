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

# ---- LOAD IMAGE ----
img_path = "data/raw/test.png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"❌ Image not found at {img_path}")

# Convert BGR → RGB (VERY IMPORTANT)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize & normalize
img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# ---- PREDICTION ----
pred = model.predict(img, verbose=0)
class_index = int(np.argmax(pred))
confidence = float(np.max(pred))

predicted_label = index_to_label[class_index]

# Safety checks
if predicted_label not in label_to_kaithi:
    raise KeyError(f"❌ Label '{predicted_label}' not in label_to_kaithi")

kaithi_char = label_to_kaithi[predicted_label]
hindi_char = map_kaithi_to_hindi(kaithi_char)

# ---- OUTPUT ----
print("Prediction confidence:", round(confidence * 100, 2), "%")
print("Predicted label:", predicted_label)
print("Kaithi:", kaithi_char)
print("Hindi:", hindi_char)


