import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\data\\final\\train"
VAL_DIR = "D:\\Projects\\Guvi_Project5\\Intelligaurd\\data\\final\\val"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

NUM_CLASSES = len(train_data.class_indices)

def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

MODELS = {
    "resnet50": ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "mobilenetv2": MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "efficientnetb0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
}

for name, base_model in MODELS.items():
    print(f"\nTraining {name}...")
    base_model.trainable = False
    model = build_model(base_model)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop])
    model.save(f"D:/Projects/Guvi_Project5/Intelligaurd/models/{name}_panel_classifier.h5")
