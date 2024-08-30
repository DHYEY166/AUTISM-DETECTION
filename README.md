## Autism Image Classification

Welcome to my Data Science Project! This project focuses on performing autism image classification using a deep learning model. The application is built with Streamlit, allowing users to upload images and receive predictions indicating whether the image is classified as Autistic or Non-Autistic.

## Table of Contents
- [App Overview](#app-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Preprocessing](#preprocessing)
- [Classes](#classes)
- [Dataset](#dataset)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## App Overview

This app allows users to upload an image for classification. The application processes the image using a pre-trained VGG-19 model and provides a prediction, indicating whether the image is classified as Autistic or Non-Autistic.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   
   git clone https://github.com/DHYEY166/AUTISM-DETECTION.git
   
   cd AUTISM-DETECTION

3. Create a virtual environment (optional but recommended):

   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. Install the required packages:

   pip install -r requirements.txt

6. Run the Streamlit application:

   streamlit run vgg19_streamlit_app.py

## Usage

- **Upload an Image**: You can upload an image in PNG, JPG, or JPEG format through the application interface.
- **Model Prediction**: Once the image is uploaded, the model will process it and display the classification result.
- **Prediction Results**: The application will show the uploaded image along with the classification result, indicating whether the image is classified as Autistic or Non-Autistic.

You can also access the application directly via the following link:

[Streamlit Application](https://autism-detection-c4eyho4yxwqsrdwh8k4gqe.streamlit.app)

## Model Details

The application utilizes the VGG-19 architecture for image classification. The model was fine-tuned using a dataset focused on autism image classification.

## Preprocessing

- **Image Resize**: All images are resized to 224x224 pixels.
- **Normalization**: The images are preprocessed using the 'preprocess_input' function from the Keras VGG-19 model, which scales pixel values to the appropriate range for the model.

## Classes

The model predicts two classes:

- **Autistic**: Indicating that the image is classified as representing an autistic individual.
- **Non-Autistic**: Indicating that the image is classified as not representing an autistic individual.

## Dataset

The model was trained using the [Autism Dataset](https://www.kaggle.com/datasets/harsh0251/autism-dataset) available on Kaggle. The dataset contains images labeled as Autistic and Non-Autistic.

## Features

- **Image Upload**: Users can upload images in PNG, JPG, or JPEG formats.
- **Model Prediction**: The model classifies the image and identifies whether it is Autistic or Non-Autistic.
- **Visualization**: The original image and the classification result are displayed for user interpretation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DHYEY166/BREAST_CANCER_SEMANTIC_SEGMENTATION/blob/main/LICENSE) file for more details.

## Contact

- **Author**: Dhyey Desai
- **Email**: dhyeydes@usc.edu
- **GitHub**: https://github.com/DHYEY166
- **LinkedIn**: https://www.linkedin.com/in/dhyey-desai-80659a216 

Feel free to reach out if you have any questions or suggestions.
