from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

IMG_SIZE = (64, 64)
BATCH = 32

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

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(64,64,3))
for layer in base.layers:
    layer.trainable = False

x = Flatten()(base.output)
x = Dense(256, activation="relu")(x)
out = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train, validation_data=val, epochs=10)

model.save("models/kaithi_ocr.h5")

