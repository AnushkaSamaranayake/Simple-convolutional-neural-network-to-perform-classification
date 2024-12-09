import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# Load the CIFAR-10 dataset
cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize the images (scale pixel values to 0-1)
x_valid, x_train = train_images[:5000] / 255.0, train_images[5000:] / 255.0
y_valid, y_train = train_labels[:5000], train_labels[5000:]
x_test = test_images / 255.0

# Class names for visualization
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

# Display a few sample images (optional)
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(train_images[i])
    axes[i].set_title(class_names[train_labels[i][0]])
    axes[i].axis('off')
plt.show()

# Build the CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout for regularization
    keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Print model summary
model.summary()

# Compile the model with Adam optimizer and learning rate 0.001
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Set up early stopping and learning rate scheduler
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=50, validation_data=(x_valid, y_valid),
                    callbacks=[early_stopping, lr_scheduler])

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # Set y-axis limits for better visualization
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, test_labels)
print(f'Test accuracy: {test_accuracy:.2f}')

# Save the trained model
model.save("cifar10_cnn_model.h5")

# Load and predict using the saved model (optional)
loaded_model = keras.models.load_model("cifar10_cnn_model.h5")
predictions = loaded_model.predict(x_test[:10])

# Display the first 10 test images with predictions
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(test_images[i])
    predicted_label = class_names[predictions[i].argmax()]
    axes[i].set_title(predicted_label)
    axes[i].axis('off')
plt.show()
