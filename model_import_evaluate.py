import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#load model
model = keras.models.load_model("mnist_model.h5")

# Evaluate the model
print(model.evaluate(test_images / 255.0, test_labels))

#comfusion matrix vis using seaborn

import seaborn as sns

confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=model.predict(test_images / 255.0).argmax(axis=1))

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
