import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1) Load & normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N,28,28,1)
x_test  = (x_test.astype("float32")  / 255.0)[..., None]

# 2) Model: small but strong CNN (~99% on MNIST)
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=128,
          validation_split=0.1, verbose=2)

# 3) Evaluate & save
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

model.save("mnist_model.keras")   # keep .h5 for widest compatibility
print("Saved model -> mnist_model.keras")
