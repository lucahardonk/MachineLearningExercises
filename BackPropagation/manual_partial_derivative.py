import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # Reshape to add channel dimension

# One-hot encode the test labels
y_test_one_hot = tf.one_hot(y_test, depth=10)

# Load the model
myModel = 'my_handwritten_model'  # saved in this directory
model = tf.keras.models.load_model(myModel)

# Define the loss function
loss_function = tf.keras.losses.MeanSquaredError()

# Evaluate the model on the test dataset before modification
y_pred_original = model.predict(x_test)
loss_original = loss_function(y_test_one_hot, y_pred_original).numpy()
print(f"Original Calculated Loss: {loss_original}")

# Choose the layer index and neuron index you want to modify
layer_index = 3  # Change this to the desired layer index
neuron_index = 0  # Change this to the desired neuron index
weight_index = 0  # Change this to the desired weight index

# Get the weights of the chosen layer
weights, biases = model.layers[layer_index].get_weights()

# Extract the weight of the chosen neuron
original_weight = weights[weight_index, neuron_index]

# Define function to calculate the loss after modifying the weight by h
def calculate_modified_loss(h):
    # Modify the weight
    modified_weights = np.copy(weights)
    modified_weights[weight_index, neuron_index] += h

    # Set the modified weights back to the layer
    model.layers[layer_index].set_weights([modified_weights, biases])

    # Predict with the modified model
    y_pred_modified = model.predict(x_test)

    # Calculate the loss with the modified predictions
    loss_modified = loss_function(y_test_one_hot, y_pred_modified).numpy()
    return loss_modified

# Calculate the initial partial derivative using TensorFlow
with tf.GradientTape() as tape:
    tape.watch(model.layers[layer_index].trainable_variables[0])
    y_pred = model(x_test, training=False)
    loss_value = loss_function(y_test_one_hot, y_pred)
grads = tape.gradient(loss_value, model.layers[layer_index].trainable_variables[0])
initial_partial_derivative = grads[weight_index, neuron_index].numpy()
print(f"Initial Partial Derivative: {initial_partial_derivative}")

# Values of h to test
h_values = [0.1, 0.05, 0.025, 0.001,0.00001, 0.000001]
losses_modified = []
partial_derivatives = []

# Calculate the partial derivatives for each h
for h in h_values:
    loss_modified = calculate_modified_loss(h)
    losses_modified.append(loss_modified)
    partial_derivative = (loss_modified - loss_original) / h
    partial_derivatives.append(partial_derivative)
    print(f"h: {h}, Loss Modified: {loss_modified}, Partial Derivative: {partial_derivative}")

# Plot the modified losses
plt.figure(figsize=(10, 6))
plt.plot(h_values, losses_modified, marker='o', linestyle='-', color='b', label='Modified Loss')
plt.axhline(y=loss_original, color='r', linestyle='--', label='Original Loss')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Loss')
plt.title('Modified Loss vs h')
plt.legend()
plt.grid(True)
plt.show()

# Plot the partial derivatives
plt.figure(figsize=(10, 6))
plt.plot(h_values, partial_derivatives, marker='o', linestyle='-', color='g', label='Partial Derivative')
plt.axhline(y=initial_partial_derivative, color='r', linestyle='--', label='Initial Partial Derivative')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Partial Derivative')
plt.title('Partial Derivative vs h')
plt.legend()
plt.grid(True)
plt.show()

# Reset the weight to the original value
weights[weight_index, neuron_index] = original_weight
model.layers[layer_index].set_weights([weights, biases])
