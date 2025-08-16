import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Ask user for image path
image_path = input("Enter the path to the digit image: ")

# Read in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Could not read image. Please check the path.")
    exit()

# Invert colors if background is white (so digit is white on black)
if np.mean(img) > 127:
    img = 255 - img

# Resize to MNIST size
img = cv2.resize(img, (28, 28))

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 28, 28, 1)

# Predict
pred = model.predict(img)
predicted_digit = np.argmax(pred)
confidence = np.max(pred) * 100

print(f"Predicted digit: {predicted_digit} ({confidence:.2f}% confidence)")
