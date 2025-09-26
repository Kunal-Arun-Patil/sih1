import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model (you will train or fine-tune one separately)
model = load_model("breed_classifier.h5")  # saved model file

# Define breed labels (example list)
breed_labels = ["Sahiwal", "Gir", "Red Sindhi", "Tharparkar", 
                "Murrah", "Mehsana", "Surti", "Jaffarabadi"]

# Streamlit UI
st.title("🐄 Cattle & Buffalo Breed Classifier")
st.write("Upload an image to identify the breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # depends on your model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = breed_labels[np.argmax(predictions)]

    st.success(f"Predicted Breed: {predicted_class}")