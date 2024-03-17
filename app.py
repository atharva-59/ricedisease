%%writefile app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/content/my_model2.hdf5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Rice Disease Detection
         """)

file = st.file_uploader("Please upload an image of a rice leaf", type=["jpg", "png"])

def preprocess_image(image):
    size = (180, 180)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = image[np.newaxis, ...]
    return img_reshape

def predict_disease(image, model):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    predictions = predict_disease(image, model)
    disease_labels = ['Leaf Blast', 'Leaf Blight', 'Tungro', 'Brown Spot']
    predicted_class_index = np.argmax(predictions)
    predicted_class = disease_labels[predicted_class_index]
    confidence = np.max(predictions)

    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)
