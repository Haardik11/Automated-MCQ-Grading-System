import os
import cv2
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, StringVar, Label, Button, Radiobutton, messagebox
import tensorflow as tf

from bubble_detection import detect_bubbles

# Load the CNN model
cnn_model_path = "cnn_model.h5" 
cnn_model = tf.keras.models.load_model(cnn_model_path)


class_labels = ["confirmed", "crossedout", "empty"]


def preprocess_image(image):
    """Preprocess an image for CNN prediction."""
    img_size = (128, 128)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    return image.reshape(1, img_size[0], img_size[1], 3)



# def preprocess_image(image):
#     """Preprocess an image for CNN prediction."""
#     if image is None or image.size == 0:
#         raise ValueError("Input image is empty or None.")
#     img_size = (128, 128)
#     image = cv2.resize(image, img_size)
#     image = image / 255.0
#     return image.reshape(1, img_size[0], img_size[1], 3)



def classify_box(image, model):
    """Classify a single box using the CNN model."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image, verbose=0)
    class_index = tf.argmax(prediction[0]).numpy()
    return class_labels[class_index]


def classify_boxes_batch(boxes, model):
    """Classify multiple boxes in a single predict() call instead of one per box."""
    img_size = (128, 128)
    batch = np.stack([cv2.resize(box, img_size) / 255.0 for box in boxes])
    predictions = model.predict(batch, verbose=0)
    class_indices = np.argmax(predictions, axis=1)
    return [class_labels[i] for i in class_indices]


def generate_model_metadata(image_path, metadata_folder):
    """Generate metadata for the model answer sheet.

    Bubble positions are detected per-image (see bubble_detection.py)
    instead of assumed from fixed pixel coordinates, so metadata stores
    which OPTION INDEX is correct per question rather than raw pixel boxes
    -- the answer key's own pixel positions aren't assumed to transfer
    directly to student sheets, which may be scanned slightly differently.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")

    questions = detect_bubbles(image)
    metadata = {"questions": []}

    for options in questions:
        boxes = [image[y:y + h, x:x + w] for (x, y, w, h) in options]
        predictions = classify_boxes_batch(boxes, cnn_model)

        correct_option = None
        for idx, prediction in enumerate(predictions):
            if prediction == "confirmed":
                correct_option = idx

        if correct_option is None:
            correct_option = 0  # Default to the first option

        metadata["questions"].append({"correct_option": correct_option})

    # Save metadata
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_path = os.path.join(metadata_folder, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path


def grade_student_folder(student_folder_path, metadata_path, output_format, output_path):
    """Grade all student answer sheets in a folder."""
    with open(metadata_path, "r") as f:
        model_metadata = json.load(f)

    results = []
    correct_options = [q["correct_option"] for q in model_metadata["questions"]]
    n_questions = len(correct_options)

    for filename in os.listdir(student_folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            student_path = os.path.join(student_folder_path, filename)
            image = cv2.imread(student_path)

            # Detect this sheet's own bubble positions instead of assuming
            # the answer key's pixel coordinates apply to every scan; a
            # sheet whose bubbles can't be confidently located is flagged
            # for manual review instead of silently graded against the
            # wrong region.
            try:
                questions = detect_bubbles(image, n_questions=n_questions)
            except ValueError as e:
                results.append({
                    "Student": filename,
                    "Score": None,
                    "Out of": n_questions,
                    "Percentage": None,
                    "Error": f"Bubble detection failed: {e}",
                })
                continue

            # Read the sheet once and classify every bubble on it in a single
            # batched predict() call, instead of re-reading the image from
            # disk and calling predict() separately for each of the (up to)
            # 12 bubbles.
            boxes = []
            for options in questions:
                for (x, y, w, h) in options:
                    boxes.append(image[y:y + h, x:x + w])
            predictions = classify_boxes_batch(boxes, cnn_model)

            score = 0
            pred_idx = 0
            for qi in range(n_questions):
                confirmed_option = None
                for oi in range(len(questions[qi])):
                    prediction = predictions[pred_idx]
                    pred_idx += 1
                    if prediction == "confirmed" and confirmed_option is None:
                        confirmed_option = oi
                if confirmed_option == correct_options[qi]:
                    score += 1
            results.append({
                "Student": filename,
                "Score": score,
                "Out of": n_questions,
                "Percentage": (score / n_questions) * 100,
                "Error": None,
            })

    # Save results
    df = pd.DataFrame(results)
    if output_format == "csv":
        df.to_csv(output_path, index=False)
    elif output_format == "xlsx":
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported format. Use csv or xlsx.")


# Tkinter GUI
root = tk.Tk()
root.title("Answer Sheet Grading App")

# Variables
model_answer_path = None
metadata_path = None
student_folder = None
output_format = StringVar(value="csv")
output_file = None


def select_model_answer():
    global model_answer_path
    model_answer_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    model_answer_label.config(text=model_answer_path if model_answer_path else "No file selected")


def generate_metadata_button():
    """Generate metadata for the model answer sheet."""
    global model_answer_path, metadata_path
    if not model_answer_path:
        messagebox.showerror("Error", "Please select a model answer sheet first!")
        return
    metadata_folder = os.path.join(os.getcwd(), "metadata")
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_path = os.path.join(metadata_folder, "model_metadata.json")
    try:
        metadata_path = generate_model_metadata(model_answer_path, metadata_folder)
        generate_metadata_label.config(text=f"Metadata saved: {metadata_path}", fg="green")
    except Exception as e:
        generate_metadata_label.config(text="Metadata generation failed!", fg="red")
        print(f"Error: {e}")


def select_metadata():
    global metadata_path
    metadata_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    load_metadata_label.config(text=metadata_path if metadata_path else "No file selected")


def select_student_folder():
    global student_folder
    student_folder = filedialog.askdirectory()
    student_folder_label.config(text=student_folder if student_folder else "No folder selected")


def save_output_file():
    global output_file
    ext = output_format.get()
    output_file = filedialog.asksaveasfilename(defaultextension=f".{ext}")
    output_file_label.config(text=output_file if output_file else "No file selected")


def run_grading():
    if not all([metadata_path, student_folder, output_file]):
        messagebox.showerror("Error", "Please ensure all inputs are selected!")
        return
    try:
        grade_student_folder(student_folder, metadata_path, output_format.get(), output_file)
        messagebox.showinfo("Success", f"Grading completed! Results saved to: {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"Grading failed: {e}")


# GUI Layout
Button(root, text="Load Model Answer Sheet", command=select_model_answer).pack(pady=5)
model_answer_label = Label(root, text="No file selected", fg="gray")
model_answer_label.pack()

Button(root, text="Generate Metadata", command=generate_metadata_button).pack(pady=5)
generate_metadata_label = Label(root, text="No metadata generated", fg="gray")
generate_metadata_label.pack()

Button(root, text="Load Metadata File", command=select_metadata).pack(pady=5)
load_metadata_label = Label(root, text="No file selected", fg="gray")
load_metadata_label.pack()

Button(root, text="Select Student Folder", command=select_student_folder).pack(pady=5)
student_folder_label = Label(root, text="No folder selected", fg="gray")
student_folder_label.pack()

Label(root, text="Select Output Format:").pack(pady=5)
Radiobutton(root, text="CSV", variable=output_format, value="csv").pack()
Radiobutton(root, text="XLSX", variable=output_format, value="xlsx").pack()

Button(root, text="Save Output File", command=save_output_file).pack(pady=5)
output_file_label = Label(root, text="No file selected", fg="gray")
output_file_label.pack()

Button(root, text="Run Grading", command=run_grading, bg="green", fg="white").pack(pady=20)

root.mainloop()



