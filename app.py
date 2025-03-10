import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Set page config
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("pneumonia_detector_latest.h5")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Custom CSS for the Light Theme with All Grey Text
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        body, .stApp {
            background-color: #f9f9ff;
            color: #666666 !important; /* Make all text grey */
        }

        /* Title */
        .title {
            font-family: 'Roboto', sans-serif;
            font-size: 44px;
            font-weight: 700;
            text-align: center;
            color: #666666 !important;
            margin-bottom: 5px;
        }

        /* Subtitle */
        .subtitle {
            font-family: 'Roboto', sans-serif;
            font-size: 18px;
            text-align: center;
            color: #666666 !important;
            margin-bottom: 30px;
        }

        /* Upload Box */
        .stFileUploader {
            border-radius: 10px !important;
            background-color: #ffffff !important;
            border: 2px solid #d4d4f7 !important;
            color: #666666 !important;
            padding: 15px !important;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #ebeaff;
            color: #666666 !important;
            font-size: 18px;
            padding: 12px 25px;
            border-radius: 8px;
            border: 1px solid #d4d4f7;
            cursor: pointer;
        }
        
        .stButton>button:hover {
            background-color: #dad5ff;
            transform: scale(1.05);
            transition: all 0.3s ease-in-out;
        }

        /* Results Styling */
        .result-box {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            margin-top: 20px;
            color: #666666 !important;
        }

        .pneumonia {
            background-color: #ffcccc; /* Light Red */
            border: 1px solid #ff9999;
        }

        .normal {
            background-color: #ccffcc; /* Light Green */
            border: 1px solid #99ff99;
        }

    </style>
""", unsafe_allow_html=True)

# ✅ Display Title & Subtitle
st.markdown("<h1 class='title'>🩺 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a Chest X-ray image and let AI detect if it's pneumonic or normal.</p>", unsafe_allow_html=True)

# ✅ Upload Section
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["png", "jpg", "jpeg"])

# ✅ Image Display & Prediction Section
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Layout for Image & Analysis
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            st.info("🔍 Analyzing the X-ray...")

            if model is not None:
                # Process the image
                processed_image = preprocess_image(image)

                # Debugging: Check input shape
                st.write(f"🖼 Processed Image Shape: {processed_image.shape}")  

                # Make prediction
                prediction = model.predict(processed_image)

                # Ensure correct indexing based on model output shape
                confidence = float(prediction[0])  # If model has single neuron output

                # Define class labels
                predicted_class = "PNEUMONIA" if confidence > 0.5 else "NORMAL"

                # Display results
                if confidence > 0.5:
                    st.markdown(f"<div class='result-box pneumonia'>⚠️ Pneumonia Detected (Confidence: {confidence:.2%})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box normal'>✅ No Pneumonia Detected (Confidence: {1 - confidence:.2%})</div>", unsafe_allow_html=True)
            else:
                st.error("⚠️ Model is not loaded. Please check your model file.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# ✅ Custom Button Styling
st.markdown(
    "<style>div.stButton > button {background-color: #ebeaff; color: #666666 !important; padding: 12px 25px; border-radius: 8px; font-size: 16px;}</style>",
    unsafe_allow_html=True
)
