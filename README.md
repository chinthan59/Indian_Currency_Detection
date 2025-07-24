# 🇮🇳 Indian Currency Recognition using Deep Learning

This project identifies Indian currency notes from uploaded images using a Convolutional Neural Network with MobileNetV2 and provides a user-friendly interface through Streamlit. It also includes text-to-speech functionality to announce the prediction.

---

## 📌 Features

- 📷 Upload an image of an Indian currency note
- 🧠 Predicts denomination using a MobileNetV2-based CNN
- 🗣️ Speaks out the result using text-to-speech (pyttsx3)
- 🚨 Warns if the uploaded note is not an Indian currency

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, PIL, OpenCV
- pyttsx3 (for audio output)

---

## 🧪 Model Training

The model was trained using **transfer learning** on a custom dataset of Indian currency notes, categorized by denomination.

### 📂 Dataset Structure
Indian currency dataset v1/
├── training/
│ ├── 10/
│ ├── 20/
│ ├── 50/
│ ├── 100/
│ ├── 200/
│ ├── 500/
│ ├── 2000/
│ └── Not Indian Currency/
├── validation/
└── test/

## 🧪 How to get started
-clone the repository
-import the dataset and save it in project folder
-install all the requirements
-run "streamlit app.py" 
-good to go
