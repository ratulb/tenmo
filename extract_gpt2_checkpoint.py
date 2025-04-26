import tensorflow as tf
import numpy as np

# Path to the GPT-2 model checkpoint directory
checkpoint_path = "models/124M"  # Change this if needed

# Load the checkpoint
reader = tf.train.load_checkpoint(checkpoint_path)

# Dictionary to store weights
weights = {}

# Iterate over all variables in the checkpoint
for name in reader.get_variable_to_shape_map():
    print(f"Loading: {name}")
    tensor = reader.get_tensor(name)
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    print(tensor.shape)
    weights[name] = tensor

# Save all the weights to a .npz file
np.savez("gpt2_weights.npz", **weights)

print("✅ GPT-2 weights extracted to gpt2_weights.npz")

