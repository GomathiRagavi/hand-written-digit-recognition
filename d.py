import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_model.keras")

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.markdown(
    "<h1 style='text-align: center; font-size: 40px;'>🖌 Handwritten Digit Recognizer</h1>",
    unsafe_allow_html=True
)

# Upload image option
uploaded_file = st.file_uploader("📂 Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Drawing canvas
st.write("Or draw a digit below:")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

def preprocess_image(img):
    img = img.convert("L")  # Grayscale
    img = ImageOps.invert(img)  # Invert colors
    img = img.resize((28, 28))  # Resize to MNIST size
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)
    return img

if st.button("🔍 Predict Digit", use_container_width=True):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = preprocess_image(img)
        pred = np.argmax(model.predict(img))
        st.success(f"✅ Predicted Digit: {pred}")

    elif canvas_result.image_data is not None:
        # Check if canvas is empty (all white)
        if np.all(canvas_result.image_data[:, :, :3] == 255):
            st.warning("⚠ Please draw or upload image before predicting .")
        else:
            img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
            img = preprocess_image(img)
            pred = np.argmax(model.predict(img))
            st.success(f"✅ Predicted Digit: {pred}")
    else:
        st.warning("⚠ Please upload an image or draw a digit.")
