
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data_dir = 'C:/Users/DELL/Desktop/BankNote_Detection/Indian currency dataset v1'
img_size = 224

train_dir = f'{data_dir}/training'
val_dir = f'{data_dir}/validation'
test_dir = f'{data_dir}/test'

## Step 3: Data Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(img_size, img_size),
                                           class_mode='categorical', batch_size=32)
val_data = val_gen.flow_from_directory(val_dir, target_size=(img_size, img_size),
                                       class_mode='categorical', batch_size=32)
test_data = test_gen.flow_from_directory(test_dir, target_size=(img_size, img_size),
                                         class_mode='categorical', batch_size=32)

num_classes = len(train_data.class_indices)

## Step 4: Model with Transfer Learning
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Step 5: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

## Step 6: Training
history = model.fit(train_data, validation_data=val_data, epochs=15, callbacks=[early_stop, checkpoint])

## Step 7: Evaluation
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")

## Step 8: Save model
model.save("currency_model_mobilenetv2.h5")
