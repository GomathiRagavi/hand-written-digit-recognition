🖌️ Handwritten Digit Recognition Web App

A simple yet powerful web application that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The app allows users to either **draw digits on a canvas** or **upload an image**, and then predicts the digit in real time.

🚀 Features
- Dual Input Options
  - 🖌️ Draw digits on a digital canvas  
  - 📂 Upload digit images (PNG/JPG)

  Real-Time Prediction
  - Classifies digits (0–9) instantly with high accuracy

  Error Handling 
  - Shows a **warning message** if no digit is drawn or uploaded

  Interactive UI  
  - Built with Streamlit and optimized with a bigger canvas for easy drawing

🛠️ Tech Stack
Python 
TensorFlow / Keras – for training and loading the CNN model  
NumPy & Pillow (PIL)– for image preprocessing (grayscale, resizing, normalization)  
Streamlit– for building the web app UI  
streamlit-drawable-canvas – for freehand drawing support  

Run the app -- streamlit run d.py

Open in browser
The app will run locally 

📊 Model Details

Dataset: MNIST (70,000 handwritten digits)
Model: CNN (Convolutional Neural Network)
Input size: 28×28 grayscale images
Accuracy: ~99% on training, ~98% on test set
