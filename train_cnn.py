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

# Mild augmentation on the train set only: bubble crops are small and the
# crossedout/confirmed marks are the whole signal, so keep shifts/rotation
# small enough not to crop the mark out of frame or flip it into a
# different-looking shape.
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=8,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.08,
    brightness_range=(0.85, 1.15),
)
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

# CNN model definition: dropout to reduce overfitting to the training crops
# (the baseline model overfit and specifically confused crossedout->confirmed
# on held-out data). BatchNormalization was tried here first but combined
# with Adam's default learning rate it made training unstable (val_loss
# spiked as high as 38 between epochs) and early-stopped on a worse model
# than the un-normalized baseline, so it was dropped in favor of dropout only.
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes: confirmed, crossedout, empty
])

# Compile the model. Lower learning rate than Adam's 1e-3 default for
# steadier convergence on this dataset.
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model. Early stopping picks the best-val-accuracy epoch instead
# of a fixed epoch count, and restores those weights before saving.
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
]

cnn_history = cnn_model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    callbacks=callbacks,
)

# Save the trained CNN model
#
# CAUTION before overwriting the shipped cnn_model.h5 with a fresh run of
# this script: prepared_dataset/{train,test} is the only labeled data this
# project has, and it does not fully represent production input. A retrain
# with this exact config raised held-out prepared_dataset/test accuracy from
# 91.08% to 93.24%, but on the real scanned sheets in StudentAnswerSheets/ it
# started misreading several genuinely-confirmed bubbles as crossedout -- a
# regression invisible to the prepared_dataset metric. Validate any retrained
# model against real scanned sheets (not just prepared_dataset/test) before
# replacing the shipped one.
cnn_model_path = "cnn_model.h5"  # Update the path if needed
cnn_model.save(cnn_model_path)

print(f"Model saved at: {cnn_model_path}")
