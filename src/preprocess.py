import cv2
import os

INPUT_DIR = "./data/raw"
OUTPUT_DIR = "./data/processed"


os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, img_name), 0)
    img = cv2.resize(img, (64, 64))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
