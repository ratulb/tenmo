"""
MNIST training with SGD and the tensor-native `tenmo.DataLoader`.

Per-batch-loop counterpart of examples/mnist.mojo (same 784 -> 128 -> 32
-> 10 MLP, SGD lr=0.01, momentum=0.9, weight_decay=1e-4, clip 1.0/0.5,
LR / 10 at epoch 10) — the SGD mirror of examples/mnist_adamw.py:

  * train()  — shuffled batches gathered into a persistent buffer
  * eval()   — sequential zero-copy view slices of the source tensors

(examples/mnist.py covers the same training via the whole-epoch
`tenmo.train_epoch` helper; this file keeps the explicit per-batch loop
that the Mojo original uses.)

Usage:
    pixi run python examples/mnist_sgd.py [epochs]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-binding"))

import numpy as np
import tenmo
from tenmo import (
    Sequential,
    Linear,
    ReLU,
    CrossEntropyLoss,
    SGD,
    DataLoader,
)

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def train_mnist_sgd():
    print("=" * 80)
    print("MNIST Training with SGD (per-batch loop with tenmo.DataLoader)")
    print("=" * 80 + "\n")

    print("Loading MNIST dataset...")
    import mnist_datasets

    loader = mnist_datasets.MNISTLoader(folder="/tmp")

    train_data = loader.load()
    train_images = np.array(train_data[0], dtype=np.float32)
    train_labels = np.array(train_data[1], dtype=np.int64)

    test_data = loader.load(train=False)
    test_images = np.array(test_data[0], dtype=np.float32)
    test_labels = np.array(test_data[1], dtype=np.int64)

    # Normalize to [0, 1], then z-score with the library constants
    def normalize(x):
        return (x / 255.0 - MNIST_MEAN) / MNIST_STD

    X_train = normalize(train_images).reshape(-1, 784).astype(np.float32)
    X_test = normalize(test_images).reshape(-1, 784).astype(np.float32)
    y_train = train_labels
    y_test = test_labels

    print("Data shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_test:", X_test.shape, "  y_test:", y_test.shape, "\n")

    # ========== Model Architecture ==========
    print("Building model...")
    model = Sequential([
        Linear(784, 128, init_method="he", bias_zero=True).into(),
        ReLU().into(),
        Linear(128, 32, init_method="he", bias_zero=True).into(),
        ReLU().into(),
        Linear(32, 10, init_method="he", bias_zero=True).into(),
    ])
    print("  Architecture: 784 -> 128 -> 32 -> 10\n")

    # ========== Training Setup (SGD, matching mnist.mojo) ==========
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    batch_size = 64

    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        clip_norm=1.0,
        clip_value=0.5,
    )

    # ========== DataLoaders ==========
    train_loader = DataLoader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    eval_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)
    print("Batches per epoch:", len(train_loader), "(train),", len(eval_loader), "(eval)")
    print("Epochs:", num_epochs, "| Batch size:", batch_size, "\n")

    def evaluate():
        eval_loader.eval()
        model.eval()
        correct = total = 0
        for xb, yb in eval_loader:
            pred = np.argmax(model(xb).numpy(), axis=1)
            correct += int((pred == yb.numpy()).sum())
            total += yb.numel
        return correct / total

    print("=" * 80)
    training_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        # Learning rate decay (same schedule as the Mojo run)
        if epoch == 10:
            optimizer.set_lr(optimizer.get_lr() / 10)

        model.train()
        train_loader.train()
        running_loss = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / num_batches
        val_acc = evaluate()

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.4f}s "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Acc: {val_acc * 100:.2f}%"
        )

    total_time = time.perf_counter() - training_start
    print("=" * 80)
    print(f"Training completed in {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    train_mnist_sgd()
