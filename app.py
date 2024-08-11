# Save this code in a file named `app.py`
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load pre-trained model
model = tf.keras.models.load_model('your_model.h5')  # Replace with your model path

def preprocess_image(image):
    image = image.resize((128, 128))  # Adjust size as needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title('Doodle Beautifier')

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.image(prediction[0], caption='Beautified Image', use_column_width=True)