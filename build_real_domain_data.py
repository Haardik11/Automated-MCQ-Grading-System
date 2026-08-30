"""
Extract every bubble crop from the real scanned sheets (AnswerKey +
StudentAnswerSheets), pseudo-label them with a trusted model (a copy of
the currently-shipped cnn_model.h5, saved as cnn_model_trusted_ref.h5
before starting a retrain), and split them into a real-domain train set
(folded into prepared_dataset/train) and a held-out real-domain
validation set (prepared_dataset_real_val) used only for evaluation,
never training.

Why: prepared_dataset's train/test images don't fully represent what a
real scanned sheet looks like. A model retrained only on prepared_dataset
raised accuracy there but started misreading confirmed bubbles on the
real sheets as crossed-out. Mixing in real crops (rerun train_cnn.py
after this script) fixes that blind spot. Always check the resulting
model against prepared_dataset_real_val AND a full grading run on
StudentAnswerSheets before trusting it enough to ship.
"""
import os
import random
import cv2
import numpy as np
import tensorflow as tf

random.seed(42)

class_labels = ["confirmed", "crossedout", "empty"]
model = tf.keras.models.load_model("cnn_model_trusted_ref.h5")

question_metadata = [
    {"options": [(209, 720, 63, 45), (550, 719, 63, 45), (929, 718, 62, 44)]},
    {"options": [(209, 1235, 63, 44), (548, 1237, 64, 44), (929, 1235, 62, 45)]},
    {"options": [(209, 1490, 63, 44), (550, 1490, 63, 45), (929, 1490, 63, 45)]},
    {"options": [(209, 1870, 63, 44), (548, 1869, 63, 45), (929, 1870, 63, 45)]},
]

sheets = [("AnswerKey/modelAnswer.png", "key")]
for i in range(1, 13):
    sheets.append((f"StudentAnswerSheets/s{i:02d}.png", f"s{i:02d}"))


def classify_boxes_batch(boxes, model):
    batch = np.stack([cv2.resize(b, (128, 128)) / 255.0 for b in boxes])
    preds = model.predict(batch, verbose=0)
    return [class_labels[i] for i in np.argmax(preds, axis=1)]


crops = []  # (image, label, source_name)
for path, tag in sheets:
    image = cv2.imread(path)
    boxes = []
    coords = []
    for qi, q in enumerate(question_metadata):
        for oi, (x, y, w, h) in enumerate(q["options"]):
            boxes.append(image[y:y + h, x:x + w])
            coords.append((qi, oi))
    preds = classify_boxes_batch(boxes, model)
    for (qi, oi), box, label in zip(coords, boxes, preds):
        crops.append((box, label, f"{tag}_q{qi}_o{oi}"))

print(f"Total real crops: {len(crops)}")
counts = {c: 0 for c in class_labels}
for _, label, _ in crops:
    counts[label] += 1
print("Pseudo-label distribution:", counts)

# Stratified 80/20 split per class
by_class = {c: [] for c in class_labels}
for item in crops:
    by_class[item[1]].append(item)
for c in class_labels:
    random.shuffle(by_class[c])

train_items, val_items = [], []
for c in class_labels:
    items = by_class[c]
    n_val = max(1, int(len(items) * 0.2)) if len(items) >= 5 else 0
    val_items.extend(items[:n_val])
    train_items.extend(items[n_val:])

print(f"Real-domain train: {len(train_items)}, held-out real-domain val: {len(val_items)}")

train_dir = "prepared_dataset/train"
val_dir = "prepared_dataset_real_val"
os.makedirs(val_dir, exist_ok=True)
for c in class_labels:
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

for box, label, name in train_items:
    out_path = os.path.join(train_dir, label, f"real_{name}.png")
    cv2.imwrite(out_path, box)

for box, label, name in val_items:
    out_path = os.path.join(val_dir, label, f"real_{name}.png")
    cv2.imwrite(out_path, box)

print("Done. Real crops added to prepared_dataset/train/<class>/real_*.png")
print("Held-out real-domain validation crops saved to prepared_dataset_real_val/<class>/")
