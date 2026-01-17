# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Model

# IMG_SIZE = (64, 64)
# BATCH = 32

# train_gen = ImageDataGenerator(rescale=1./255)
# val_gen = ImageDataGenerator(rescale=1./255)

# train = train_gen.flow_from_directory(
#     "dataset/train",
#     target_size=IMG_SIZE,
#     batch_size=BATCH,
#     class_mode="categorical"
# )

# val = val_gen.flow_from_directory(
#     "dataset/val",
#     target_size=IMG_SIZE,
#     batch_size=BATCH,
#     class_mode="categorical"
# )

# base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(64,64,3))
# for layer in base.layers:
#     layer.trainable = False

# x = Flatten()(base.output)
# x = Dense(256, activation="relu")(x)
# out = Dense(train.num_classes, activation="softmax")(x)

# model = Model(base.input, out)
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# model.fit(train, validation_data=val, epochs=10)

# model.save("models/kaithi_ocr.h5")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import json
import os

IMG_SIZE = (64, 64)
BATCH = 32

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

val = val_gen.flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

# Save class indices (VERY IMPORTANT)
with open("models/class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train.class_indices, f, ensure_ascii=False, indent=4)

# Load base model (Transfer Learning)
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(64, 64, 3)
)

# Freeze base layers
for layer in base.layers:
    layer.trainable = False

# Custom classifier head
x = Flatten()(base.output)
x = Dense(256, activation="relu")(x)
out = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, out)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    train,
    validation_data=val,
    epochs=10
)

# Save model
model.save("models/kaithi_ocr.h5")

print("✅ Model saved as models/kaithi_ocr.h5")
print("✅ Class indices saved as models/class_indices.json")
