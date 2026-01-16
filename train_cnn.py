import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

import PIL
import os

# Paths to the prepared dataset
train_dir = "prepared_dataset/train"  # Update this to your train directory path
test_dir = "prepared_dataset/test"  # Update this to your test directory path

# Image size and batch configuration
img_size = (128, 128)
batch_size = 32

# Image data generators for CNN
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN model definition
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: confirmed, crossedout, empty
])

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
cnn_history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the trained CNN model
cnn_model_path = "cnn_model.h5"  # Update the path if needed
cnn_model.save(cnn_model_path)

print(f"Model saved at: {cnn_model_path}")
