import os
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import joblib

st.set_page_config(layout="wide")


def path_getter():
    knn_model_path = os.path.join(os.getcwd(), "model", "knn_model_7.pkl")
    cnn_model_path = os.path.join(os.getcwd(), "model", "cnn_model.h5")
    return knn_model_path, cnn_model_path


@st.cache_resource
def models_loader():
    knn_model_path, cnn_model_path = path_getter()
    knn_model = joblib.load(knn_model_path)
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    return knn_model, cnn_model


def find_image_bounds(image_array):
    non_empty_columns = np.nonzero(image_array.max(axis=0) > 0)[0]
    non_empty_rows = np.nonzero(image_array.max(axis=1) > 0)[0]
    if len(non_empty_rows) == 0 or len(non_empty_columns) == 0:
        return None
    upper_bound = non_empty_rows[0]
    lower_bound = non_empty_rows[-1]
    left_bound = non_empty_columns[0]
    right_bound = non_empty_columns[-1]
    return (left_bound, upper_bound, right_bound, lower_bound)


def image_adjustments(image_adjust):
    image_gray = image_adjust.convert("L")
    image_array = np.array(image_gray)
    margin = 50

    bounds = find_image_bounds(image_array)
    if bounds:
        image_cropped = image_gray.crop(bounds)
        image_with_margin = ImageOps.expand(image_cropped, border=margin, fill="black")
    else:
        image_with_margin = image_gray

    image_padded = ImageOps.pad(image_with_margin, (28, 28), method=0, centering=(0.5, 0.5))
    image_normalized_array = np.array(image_padded) / 255.0
    image_flatten = image_normalized_array.reshape(1, -1)
    image_dimension = image_normalized_array.reshape(1, 28, 28, 1)
    return image_flatten, image_dimension


def predict_image(image):
    knn_model, cnn_model = models_loader()
    image_from_array = Image.fromarray(np.uint8(image))
    image_flattened, image_dimension = image_adjustments(image_from_array)
    predict_label_knn = knn_model.predict(image_flattened)
    predict_label_knn_adjusted = predict_label_knn[0]
    predict_label_cnn = cnn_model.predict(image_dimension)
    return predict_label_knn_adjusted, predict_label_cnn


def start_predict(image_data):
    predict_label_knn, predict_label_cnn = predict_image(image_data)
    predict_label_cnn_index = np.argmax(predict_label_cnn)
    predict_label_cnn_prob = round(predict_label_cnn[0][predict_label_cnn_index] * 100, 2)
    st.title("Predictions :dart:")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("CNN (Convolutional Neural Network)")
        st.write(f"The image represents the number: {predict_label_cnn_index}")
        st.write(f"Probability: {predict_label_cnn_prob}%")
    with col2:
        st.subheader("KNN (K-Nearest Neighbors)")
        st.write(f"The image represents the number: {predict_label_knn}")


st.title("Handwritten Digit Recognition :pencil:")
drawing_mode = "freedraw"
stroke_width = 12
stroke_color = "#eee"
bg_color = "#000000"

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=None,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
)

if st.button("Predict!"):
    if canvas_result.image_data is not None and canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        if len(objects) > 0:
            image_data = canvas_result.image_data
            with st.spinner("Processing image and making predictions."):
                start_predict(image_data)
        else:
            st.warning("Please draw a number before pressing.")
