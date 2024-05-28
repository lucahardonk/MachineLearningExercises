import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

num_epochs = 5
batch_size = 200

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mean_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_losses.append(np.mean(self.losses))

    

# Create an instance of the custom callback
loss_history = LossHistory()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(400, activation='relu'))
model.add(tf.keras.layers.Dense(250, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.21, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[loss_history])
except KeyboardInterrupt:
    print("Training interrupted by user.")

# Save the model
model.save('my_handwritten_model')

# Model evaluation using the reserved testing data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}\n")
print(f"Test loss: {loss}")

# Plot the loss after each batch
plt.plot(loss_history.losses)
plt.title('Loss after each batch')
plt.ylabel('Loss')
plt.xlabel('Batch')

# Add vertical lines to separate epochs
for i in range(1, num_epochs):
    if i == num_epochs - 1:
        plt.axvline(x=i * len(x_train) / batch_size, color='y', linestyle='--', linewidth=0.5, label='Epochs')
    else:
        plt.axvline(x=i * len(x_train) / batch_size, color='y', linestyle='--', linewidth=0.5)


# Calculate mean loss for each epoch
loss_per_epoch = np.mean(np.array_split(loss_history.mean_losses, len(loss_history.mean_losses) // (len(x_train) // batch_size)), axis=1)

# Find the epoch index with the lowest mean loss
min_mean_loss_epoch_index = np.argmin(loss_per_epoch)
min_mean_loss_epoch = loss_per_epoch[min_mean_loss_epoch_index]

print(f"Epoch with the lowest mean loss: Epoch {min_mean_loss_epoch_index + 1}, Mean Loss: {min_mean_loss_epoch}")

# Add a horizontal line for the epoch with the lowest mean loss
#plt.axhline(y=min_mean_loss_epoch, color='r', linestyle='--', linewidth=1, label=f'Lowest Mean Loss: {min_mean_loss_epoch:.4f}')
plt.legend()

plt.show()
