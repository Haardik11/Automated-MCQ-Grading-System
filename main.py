import os
import cv2
import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog, StringVar, Label, Button, Radiobutton, messagebox
import tensorflow as tf

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
    prediction = model.predict(preprocessed_image)
    class_index = tf.argmax(prediction[0]).numpy()
    return class_labels[class_index]


def generate_model_metadata(image_path, metadata_folder):
    """Generate metadata for the model answer sheet."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")

    metadata = {"questions": []}
    confirmed_boxes = []

    # Define coordinates for each question's options
    question_metadata = [
        # Question 1
        {"options": [(209, 720, 63, 45), (550, 719, 63, 45), (929, 718, 62, 44)]},
        # Question 2
        {"options": [(209, 1235, 63, 44), (548, 1237, 64, 44), (929, 1235, 62, 45)]},
        # Question 3
        {"options": [(209, 1490, 63, 44), (550, 1490, 63, 45), (929, 1490, 63, 45)]},
        # Question 4
        {"options": [(209, 1870, 63, 44), (548, 1869, 63, 45), (929, 1870, 63, 45)]},
    ]

    for question_num, question in enumerate(question_metadata, start=1):
        options = question["options"]
        question_data = {"options": options, "confirmed": None}

        # Extract and classify each option
        for option in options:
            x, y, w, h = option
            box = image[y:y + h, x:x + w]
            prediction = classify_box(box, cnn_model)

            # If this option is confirmed, store it
            if prediction == "confirmed":
                question_data["confirmed"] = option

        # Append question data to metadata
        if question_data["confirmed"] is None:
            question_data["confirmed"] = options[0]  # Default to the first option
        confirmed_boxes.append(question_data["confirmed"])
        metadata["questions"].append(question_data)

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
    question_metadata = model_metadata["questions"]
    model_confirmed_boxes = [q["confirmed"] for q in question_metadata]

    for filename in os.listdir(student_folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            student_path = os.path.join(student_folder_path, filename)
            score, matched = 0, []
            for idx, question in enumerate(question_metadata):
                options = question["options"]
                confirmed = None
                for option in options:
                    x, y, w, h = option
                    box = cv2.imread(student_path)[y:y + h, x:x + w]
                    prediction = classify_box(box, cnn_model)
                    if prediction == "confirmed":
                        confirmed = option
                        break
                matched.append(model_confirmed_boxes[idx] == confirmed)
                if model_confirmed_boxes[idx] == confirmed:
                    score += 1
            results.append({
                "Student": filename,
                "Score": score,
                "Out of": len(question_metadata),
                "Percentage": (score / len(question_metadata)) * 100
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



