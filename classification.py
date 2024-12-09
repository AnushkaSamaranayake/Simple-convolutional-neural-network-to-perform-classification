import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the shape of the data
# print(train_images.shape)
# print(test_images.shape)

# Display the first 10 images
# fig, axes = plt.subplots(1, 10, figsize=(10, 3))
# for i in range(10):
#     axes[i].imshow(train_images[i], cmap='gray')
#     axes[i].axis('off')
# plt.show()

# Normalize the images
x_valid, x_train = train_images[:5000] / 255.0, train_images[5000:] / 255.0
y_valid, y_train = train_labels[:5000], train_labels[5000:]
x_test = test_images / 255.0

# Create the model
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.summary())
print(model.layers)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), batch_size=32)

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.save("mnist_model.h5")



