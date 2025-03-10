import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_detector_latest.h5")
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="wide")

st.markdown("""
    <h1 style="text-align:center; color:#0066cc;">🩺 Pneumonia Detection AI</h1>
    <p style="text-align:center;">Upload a Chest X-ray image and let AI detect if it's pneumonic or normal.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    with col2:
        st.info("Analyzing the X-ray...")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        
        if prediction > 0.5:
            st.error(f"⚠️ Pneumonia Detected (Confidence: {prediction:.2%})")
        else:
            st.success(f"✅ No Pneumonia Detected (Confidence: {1 - prediction:.2%})")

st.markdown(
    "<style>div.stButton > button {background-color: #0066cc; color: white; padding: 10px 20px; border-radius: 10px;}</style>",
    unsafe_allow_html=True
)
