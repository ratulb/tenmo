"""
MNIST Training with Python bindings for Tenmo.
Exact port of examples/mnist.mojo.

Uses tenmo.train_epoch / tenmo.eval_epoch which run the entire epoch
(Tenmo DataLoader + batching + SIMD normalization) inside Mojo —
zero per-batch tensor creation through the Python boundary.

Usage:
    pixi run python examples/mnist.py
"""

import sys
import os
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
    train_epoch,
    eval_epoch,
)

# Constants (matching tenmo/dataloader.mojo)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def train_mnist():
    """Train a neural network on MNIST dataset."""
    print("=" * 80)
    print("MNIST Training")
    print("=" * 80 + "\n")

    # ========== Data Loading ==========
    print("Loading MNIST dataset...")
    import mnist_datasets

    loader = mnist_datasets.MNISTLoader(folder="/tmp")

    train_data = loader.load()
    train_images = np.array(train_data[0], dtype=np.float32)
    train_labels = np.array(train_data[1], dtype=np.float32)

    test_data = loader.load(train=False)
    test_images = np.array(test_data[0], dtype=np.float32)
    test_labels = np.array(test_data[1], dtype=np.float32)

    print("  Train samples:", len(train_images))
    print("  Test samples:", len(test_images), "\n")

    # ========== Data Preparation ==========
    # Normalize to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Flatten 28x28 → 784
    X_train = train_images.reshape(-1, 784).astype(np.float32)
    X_test = test_images.reshape(-1, 784).astype(np.float32)
    y_train = train_labels.astype(np.float32)
    y_test = test_labels.astype(np.float32)

    print("Data shapes:")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test:", X_test.shape)
    print("  y_test:", y_test.shape, "\n")

    # ========== Model Architecture ==========
    print("Building model...")
    model = Sequential([
        Linear(784, 128, init_method="he", bias_zero=True).into(),
        ReLU().into(),
        Linear(128, 32, init_method="he", bias_zero=True).into(),
        ReLU().into(),
        Linear(32, 10, init_method="he", bias_zero=True).into(),
    ])
    print("  Architecture: 784 -> 128 -> 32 -> 10")
    print("  Total parameters:", model.num_parameters(), "\n")

    # ========== Training Setup ==========
    num_epochs = 15
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    clip_norm = 1.0
    clip_value = 0.5

    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )

    print("Training configuration:")
    print("  Epochs:", num_epochs)
    print("  Batch size: 64")
    print("  Learning rate:", learning_rate)
    print("  Momentum:", momentum, "\n")

    print("=" * 80)
    training_start = time.perf_counter()

    # ========== Training Loop ==========
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        # Learning rate decay
        if epoch == 10:
            optimizer.set_lr(optimizer.get_lr() / 10)
        if epoch == 15:
            optimizer.set_lr(optimizer.get_lr() / 10)

        # --- Training Phase (entire epoch in Mojo) ---
        train_loss, train_acc = train_epoch(
            model,
            criterion,
            optimizer,
            X_train,
            y_train,
            batch_size=64,
            shuffle=True,
            normalize_mean=MNIST_MEAN,
            normalize_std=MNIST_STD,
        )

        # --- Validation Phase (entire epoch in Mojo) ---
        val_loss, val_acc = eval_epoch(
            model,
            criterion,
            X_test,
            y_test,
            batch_size=64,
            normalize_mean=MNIST_MEAN,
            normalize_std=MNIST_STD,
        )

        # --- Epoch Report ---
        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| Time: {epoch_time:.4f}s "
            f"| Train Loss: {train_loss:.6f} Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.6f} Acc: {val_acc * 100:.2f}%"
        )

    total_time = time.perf_counter() - training_start
    print("=" * 80)
    print(f"Training completed in {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    train_mnist()
