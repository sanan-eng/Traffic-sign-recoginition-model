import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import time
model_path=r"E:\SUMMER_INTERN\traffic_sign_model_enhanced.h5"
# Set page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define class names for GTSRB dataset (43 classes)
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]


# Preprocessing function for the pre-trained model
def preprocess_image(img, target_size=(48, 48)):
    """
    Preprocess image for the pre-trained model.
    Resizes, normalizes, and prepares image for prediction.
    """
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:  # BGR (OpenCV default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    img = cv2.resize(img, target_size)

    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0

    # Expand dimensions to create batch size of 1
    img = np.expand_dims(img, axis=0)

    return img


# Load pre-trained model (using a placeholder function - in practice, you would load a real pre-trained model)
@st.cache_resource
def load_pretrained_model():
    """
    Load a pre-trained model for traffic sign recognition.
    In a real implementation, this would load a model like ResNet-34 pre-trained on GTSRB.
    """
    # For demonstration purposes, we'll create a simple model architecture
    # In practice, you would load an actual pre-trained model
    model =load_model(model_path)
    st.info("Using a placeholder model. In practice, load a model pre-trained on GTSRB like ResNet-34.")
    return model


# Function to make predictions
def predict_traffic_sign(model, img):
    """
    Predict traffic sign class from image.
    Returns class index, class name, and confidence.
    """
    # Preprocess image
    processed_img = preprocess_image(img)

    # Make prediction
    predictions = model.predict(processed_img, verbose=0)

    # Get top prediction
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = CLASS_NAMES[predicted_class]

    return predicted_class, class_name, confidence


# Main Streamlit app
def main():
    # App title and description
    st.title("🚦 Traffic Sign Recognition")
    st.markdown("""
    This app recognizes traffic signs using a pre-trained CNN model.
    Upload an image of a traffic sign, and the app will predict its class.
    """)

    # Load model
    model = load_pretrained_model()

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This app uses a pre-trained Convolutional Neural Network (CNN) 
    model to recognize traffic signs from the German Traffic Sign 
    Recognition Benchmark (GTSRB) dataset.

    The model can classify traffic signs into 43 different categories.
    """)

    st.sidebar.header("Settings")
    allow_multiple = st.sidebar.checkbox("Allow multiple file uploads", value=True)
    show_confidence = st.sidebar.checkbox("Show confidence scores", value=True)

    # File upload section
    st.header("Upload Traffic Sign Images")
    uploaded_files = st.file_uploader(
        "Choose traffic sign images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=allow_multiple
    )

    if uploaded_files:
        # Initialize results list for CSV export
        results_list = []

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Display file details
            st.markdown(f"*File:* {uploaded_file.name}")

            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

            # Make prediction
            with st.spinner("Analyzing..."):
                predicted_class, class_name, confidence = predict_traffic_sign(model, img)

            # Display prediction
            with col2:
                st.success("Prediction Result")
                st.markdown(f"*Class:* {predicted_class}")
                st.markdown(f"*Sign:* {class_name}")
                if show_confidence:
                    st.markdown(f"*Confidence:* {confidence:.2%}")

                # Color code based on confidence
                if confidence > 0.75:
                    st.success("High confidence prediction")
                elif confidence > 0.5:
                    st.warning("Moderate confidence prediction")
                else:
                    st.error("Low confidence prediction")

            # Add to results list for CSV export
            results_list.append({
                "Filename": uploaded_file.name,
                "Class_ID": predicted_class,
                "Class_Name": class_name,
                "Confidence": confidence
            })

            st.markdown("---")

        # Create results dataframe
        if results_list:
            results_df = pd.DataFrame(results_list)

            # Display results table
            st.header("Prediction Results Summary")
            st.dataframe(results_df)

            # CSV download
            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="traffic_sign_predictions.csv",
                mime="text/csv"
            )
    else:
        # Display sample images and predictions if no files uploaded
        st.info("Please upload traffic sign images to get predictions.")

        # Sample images section
        st.header("Sample Traffic Signs")
        sample_cols = st.columns(3)

        # Sample images (would need actual sample images in a real implementation)
        with sample_cols[0]:
            st.markdown("*Stop Sign*")
            st.image(np.zeros((32, 32, 3), dtype=np.uint8), caption="Sample image would appear here")

        with sample_cols[1]:
            st.markdown("*Speed Limit (50km/h)*")
            st.image(np.zeros((32, 32, 3), dtype=np.uint8), caption="Sample image would appear here")

        with sample_cols[2]:
            st.markdown("*Yield Sign*")
            st.image(np.zeros((32, 32, 3), dtype=np.uint8), caption="Sample image would appear here")


# Run the app
if __name__ == "__main__":
    main()