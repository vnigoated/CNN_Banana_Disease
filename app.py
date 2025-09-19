import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Parameters
IMG_SIZE = 150

# Load the trained model
model = tf.keras.models.load_model('banana_disease_classifier.h5')

# Mapping class indices to labels (based on training class folder names)
class_labels = [
    'black_sigatoka',
    'bract_virus',
    'healthy_leaf',
    'insect_pest',
    'moko_disease',
    'panama_disease',
    'yellow_sigatoka'
]

def predict_banana_disease(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx] * 100

    return f"{class_labels[class_idx]} ({confidence:.2f}%)"

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_banana_disease,
    inputs=gr.Image(type='pil', label="Upload Banana Image"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="🍌 Banana Disease Classifier",
    description="Upload an image of a banana leaf or fruit, and the model will classify it into one of the disease categories."
)

# Launch the UI
interface.launch()
