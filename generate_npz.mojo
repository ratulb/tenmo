### This file generates "gpt2_weights.npz" file after checkpoint files have been downloaded 
### by running the "fetch_gpt2_weights.mojo" file.

from python import Python

fn main():
    try:
        np = Python.import_module("numpy")
        tf = Python.import_module("tensorflow")
        # Load weights from the checkpoint
        reader = tf.train.load_checkpoint("models/124M")
        weights = Python.dict()
        for name in reader.get_variable_to_shape_map():
            tensor = reader.get_tensor(name)
            print(name, tensor.shape)
            weights[name] = reader.get_tensor(name)
        np.savez("gpt2_weights.npz", **weights)
    except e:
        print(e)


