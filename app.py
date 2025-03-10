import streamlit as st

# ✅ Set page config FIRST before any other Streamlit command
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("pneumonia_detector_latest.h5")  # Use latest model name
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((150, 150))  # Resize to match model training
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize (0-1 scale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.markdown("""
    <h1 style="text-align:center; color:#0066cc;">🩺 Pneumonia Detection AI</h1>
    <p style="text-align:center;">Upload a Chest X-ray image and let AI detect if it's pneumonic or normal.</p>
        <style>
        .stApp { background-color: #f0f2f6; }
        .stButton>button { background-color: #FF5733; color: white; }
        .stTextInput>div>div>input { color: #333; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

        with col2:
            st.info("Analyzing the X-ray...")

            if model is not None:
                # Process the image
                processed_image = preprocess_image(image)

                # Debugging: Check input shape
                st.write(f"Processed Image Shape: {processed_image.shape}")  

                # Make prediction
                prediction = model.predict(processed_image)

                # Ensure correct indexing based on model output shape
                confidence = float(prediction[0])  # If model has single neuron output

                # Define class labels
                predicted_class = "PNEUMONIA" if confidence > 0.5 else "NORMAL"

                # Display results
                if confidence > 0.5:
                    st.error(f"⚠️ Pneumonia Detected (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ No Pneumonia Detected (Confidence: {1 - confidence:.2%})")
            else:
                st.error("⚠️ Model is not loaded. Please check your model file.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# Custom Button Styling
st.markdown(
    "<style>div.stButton > button {background-color: #0066cc; color: white; padding: 10px 20px; border-radius: 10px;}</style>",
    unsafe_allow_html=True
)
