import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#load model
model = keras.models.load_model("mnist_model.h5")

#for loop for predict test set elements
for i in range(100):
    print("Actual: ", test_labels[i])
    print("Predicted: ", model.predict(test_images[i].reshape(1, 28, 28)).argmax())
    # plt.imshow(test_images[i], cmap='gray')
    # plt.show()
    print("--------------------------------------------------")