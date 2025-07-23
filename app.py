import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load model
model = load_model("keras_model.h5")

# Load class labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Title
st.title("Indian Currency Recognition")

uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(predictions[0][class_index])
    prediction = class_names[class_index]

    # Result
    if prediction.lower() == "not indian currency" or confidence < 0.7:
        message = "This is not an Indian currency note."
        st.warning(message)
    else:
        message = f"Predicted Denomination: ₹{prediction} Confidence: {confidence:.2f}"
        st.success(message)

    # Speak the result
    engine.say(message)
    engine.runAndWait()