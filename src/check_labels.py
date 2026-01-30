# src/check_labels.py
import json

with open("models/class_indices.json", "r", encoding="utf-8") as f:
    print(json.load(f))
