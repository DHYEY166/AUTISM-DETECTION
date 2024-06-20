# vgg19_streamlit_app.py
import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the base directory where the script and data folders are located
base_dir = os.getcwd()

# Paths to the train and test directories
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Function to create dataframes for train and test datasets
def create_dataframe(dir_path):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    filenames = os.listdir(dir_path)
    categories = [1 if 'autistic' in filename.lower() else 0 for filename in filenames]
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return df

try:
    # Create dataframes for train and test datasets
    train_df = create_dataframe(train_dir)
    test_df = create_dataframe(test_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Load VGG-19 model (ensure the correct path to your model weights file)
model_weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
absolute_model_weights_path = os.path.join(base_dir, model_weights_path)

if os.path.exists(absolute_model_weights_path):
    # Load base model
    base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights=None)
    base_model.load_weights(absolute_model_weights_path)
    
    # Build the classification head
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    st.success("Model loaded and compiled successfully.")
else:
    st.error(f"Model weights not found: {absolute_model_weights_path}")

# Streamlit Interface
st.title("Autism Image Classification")
st.write("This app classifies images as either Autistic or Non-Autistic.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = load_img(uploaded_file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Predict the class
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        st.write("The image is classified as **Autistic**.")
    else:
        st.write("The image is classified as **Non-Autistic**.")
