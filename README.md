# MCQ Answer Sheet Checking System

## Overview
The **MCQ Answer Sheet Checking System** is an automated grading tool that evaluates student answer sheets against a predefined answer key. The project leverages **Convolutional Neural Networks (CNN)** for Optical Character Recognition (OCR) to read and evaluate scanned student responses.

## Features
- **Automated Grading**: Uses CNN to recognize and compare answers.
- **Metadata Generation**: Extracts and saves relevant data for verification.
- **GUI Interface**: Simplified interface for ease of use.
- **Customizable Reports**: Generates grading reports in various formats.

## Project Structure
```
MCQ-AnswerSheet-Checker/
│── AnswerKey/             # Contains Model Answer Sheet & Marking Scheme
│── Metadata/              # Stores generated metadata JSON files
│── prepared_dataset/      # Training dataset (Train & Test folders)
│── StudentAnswerSheets/   # Scanned student answer sheets
│── train_cnn.py           # Script to train CNN model
│── cnn_model.h5           # Pre-trained CNN model
│── main.py                # Core script for GUI, metadata generation, and grading
│── requirements.txt       # List of required Python libraries
│── README.md              # Project documentation
```

## Requirements
- Python **3.6 - 3.11**
- Required Libraries:
  ```bash
  pip install opencv-python pandas tensorflow pillow openpyxl
  ```

## Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/ParasJain19/Answer-Sheet-Checking-System.git
cd MCQ-AnswerSheet-Checker
```

### 2. Install Dependencies
```bash
pip install opencv-python pandas tensorflow pillow openpyxl
```
### 3. Run the Application
```bash
python main.py
```

### 4. Steps to Grade Answer Sheets
1. Open the **MCQ Answer Sheet Checking System** GUI.
2. Load `ModelAnswer.png` from the **AnswerKey** folder.
3. Generate metadata and verify it (metadata is saved in the **Metadata** folder).
4. Load the metadata and **StudentAnswerSheets** folder.
5. Select the output report format and location.
6. Click **Start Grading**.

## Model Training
If you wish to train the model from scratch, run:
```bash
python train_cnn.py
```
This will train a new **cnn_model.h5** based on the dataset.

## Output
- The graded results are saved in the chosen output directory in CSV or PDF format.

## Contributing
Feel free to fork and contribute via pull requests.


---
### Author: PARAS JAIN


