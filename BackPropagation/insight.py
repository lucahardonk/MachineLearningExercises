import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('my_handwritten_model')

# Print model summary to identify layer names and shapes
model.summary()

# Choose the layer index and neuron index you want to visualize
layer_index = 3  #  Dense layer index
neuron_index = 9  # Choose your neuron
weight_index = 68  # choose the weight of the chosen neuron to print

# Get the weights of the chosen layer
weights, biases = model.layers[layer_index].get_weights()

# Print the shape of the weights to understand the dimensions
print(f"Shape of weights: {weights.shape}") 
# Extract the weights of the specific neuron
neuron_weights = weights[:, neuron_index]
# Print the specific weight of the choosen neuron
print(f"Weight at index {weight_index}: {neuron_weights[weight_index]}")


# Plot the weights as a 1D heatmap
plt.figure(figsize=(20, 15))
plt.imshow(neuron_weights[np.newaxis, :], aspect="auto", cmap="viridis")
plt.colorbar()
plt.title(f'Weights of Neuron {neuron_index} in Layer {layer_index}')
plt.show()
