import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('brain_tumor_model.h5')

# Define class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to detect the type of tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show results
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2%}")
