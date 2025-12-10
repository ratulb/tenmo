"""
Spiral Classification Example
===============================

This example demonstrates training a neural network on the classic spiral dataset,
a popular benchmark for testing non-linear classification capabilities.

Problem:
    Binary classification of 2D points arranged in two interleaved spirals.
    This is a challenging non-linear problem that tests a network's ability to
    learn complex decision boundaries.

Network Architecture:
    Input:  2 features (x, y coordinates)
    Hidden: 64 → 32 → 16 neurons with ReLU activation
    Output: 1 neuron with Sigmoid activation (binary classification)

Key Features Demonstrated:
    ✓ Mini-batch training with DataLoader
    ✓ Train/Validation split
    ✓ Train/Eval mode switching
    ✓ Binary Cross-Entropy loss
    ✓ Accuracy monitoring
    ✓ Proper gradient tracking control

Dataset:
    - Training: 500 samples
    - Validation: 250 samples
    - Batch size: 32 (training), 64 (validation)
    - Data augmentation: Random shuffling each epoch

Performance:
    - Training time: ~1-2 minutes on CPU (15,000 epochs)
    - Expected accuracy: >95% on validation set
    - Loss convergence: <0.05 final validation loss
"""

from tenmo import Tensor
from net import (
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    SGD,
    BCELoss,
)
from data import TensorDataset, DataLoader, Batch
from time import perf_counter_ns
from math import sqrt, cos, sin, pi
from random import randn_float64
from intarray import IntArray
from common_utils import SpiralDataGenerator, binary_accuracy


fn generate_spiral_data(
    n_points: Int = 100, n_rotations: Float64 = 3.0, noise: Float64 = 0.1
) -> Tuple[Tensor[DType.float64], Tensor[DType.float64]]:
    """Generate two intertwined spirals.

    Args:
        n_points: Points per spiral.
        n_rotations: Number of full rotations (higher = more complex).
        noise: Random noise to add (makes it harder).

    Returns:
        (X, y) where:
        - X: Shape (2*n_points, 2) - coordinates.
        - y: Shape (2*n_points, 1) - labels (0 or 1).
    """
    var start = perf_counter_ns()
    var total_points = 2 * n_points
    var X = Tensor[DType.float64].zeros(total_points, 2)
    var y = Tensor[DType.float64].zeros(total_points, 1)

    for i in range(n_points):
        # Angle increases linearly
        var theta = Float64(i) / Float64(n_points) * n_rotations * 2.0 * pi

        # Radius increases with angle (spiral out)
        var radius = Float64(i) / Float64(n_points)

        # Spiral 1 (class 0)
        var x1 = radius * cos(theta) + randn_float64() * noise
        var y1 = radius * sin(theta) + randn_float64() * noise
        X[i, 0] = x1
        X[i, 1] = y1
        y[i, 0] = 0.0

        # Spiral 2 (class 1) - rotated 180 degrees
        var x2 = radius * cos(theta + pi) + randn_float64() * noise
        var y2 = radius * sin(theta + pi) + randn_float64() * noise
        X[n_points + i, 0] = x2
        X[n_points + i, 1] = y2
        y[n_points + i, 0] = 1.0

    var end = perf_counter_ns()
    print("generate_spiral_data took: ", (end - start) / 1e9, "secs")

    return (X, y)


fn accuracy[
    dtype: DType
](pred: Tensor[dtype], target: Tensor[dtype]) -> Tuple[Int, Int]:
    var correct = 0
    var total = pred.shape()[0]

    for i in range(total):
        var predicted_class = 1 if pred[i, 0] > 0.5 else 0
        var true_class = Int(target[i, 0])
        if predicted_class == true_class:
            correct += 1
    return correct, total


fn train_spiral_classifier():
    """Train a neural network to classify spiral dataset with mini-batch training.

    This example demonstrates:
    1. DataLoader for mini-batch training
    2. Train/validation split for generalization monitoring
    3. Train/eval mode for proper gradient tracking
    4. Accuracy and loss tracking
    """

    print("=" * 80)
    print("Spiral Dataset Classification")
    print("=" * 80)
    print()

    # ============================================================================
    # Configuration
    # ============================================================================

    alias dtype = DType.float64

    var num_train_samples = 500
    var num_val_samples = 250
    var train_batch_size = 32
    var val_batch_size = 64
    var num_epochs = 5000
    var learning_rate = 0.05
    var momentum = 0.1
    var log_interval = 1000

    print("Configuration:")
    print("  Training samples:", num_train_samples)
    print("  Validation samples:", num_val_samples)
    print("  Train batch size:", train_batch_size)
    print("  Validation batch size:", val_batch_size)
    print("  Number of epochs:", num_epochs)
    print("  Learning rate:", learning_rate)
    print("  Momentum:", momentum)
    print()

    # ============================================================================
    # Data Generation and Loading
    # ============================================================================

    print("Generating spiral dataset...")

    # Generate training data
    # var (X_train, y_train) = generate_spiral_data(
    var (X_train, y_train) = SpiralDataGenerator.generate_data(
        n_points=num_train_samples, n_rotations=2.0, noise=0.01
    )

    # Generate validation data
    # var (X_val, y_val) = generate_spiral_data(
    var (X_val, y_val) = SpiralDataGenerator.generate_data(
        n_points=num_val_samples, noise=0.01, n_rotations=2.0
    )

    print("  Training set: ", num_train_samples, "samples")
    print("  Validation set:", num_val_samples, "samples")
    print()

    # Create datasets
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var val_dataset = TensorDataset[dtype](X_val, y_val)

    # Create data loaders
    var train_loader = DataLoader(
        train_dataset^,
        batch_size=train_batch_size,
        shuffle=True,  # Shuffle training data each epoch
        drop_last=False,  # Use all samples
    )

    var val_loader = DataLoader(
        val_dataset^,
        batch_size=val_batch_size,
        shuffle=False,  # No shuffling for validation
        drop_last=False,
    )

    print("Data loaders created:")
    print("  Train batches per epoch:", len(train_loader))
    print("  Validation batches per epoch:", len(val_loader))
    print()

    # ============================================================================
    # Model Architecture
    # ============================================================================

    var model = Sequential[dtype]()
    model.append(
        # Layer 1: 2 → 64 with ReLU
        Linear[dtype](in_features=2, out_features=64, xavier=False).into(),
        ReLU[dtype]().into(),
        # Layer 2: 64 → 32 with ReLU
        Linear[dtype](in_features=64, out_features=32, xavier=False).into(),
        ReLU[dtype]().into(),
        # Layer 3: 32 → 16 with ReLU
        Linear[dtype](in_features=32, out_features=16, xavier=False).into(),
        ReLU[dtype]().into(),
        # Output Layer: 16 → 1 with Sigmoid
        Linear[dtype](in_features=16, out_features=1, xavier=False).into(),
        Sigmoid[dtype]().into(),
    )

    print("Model Architecture:")
    print("  Input Layer:     2 features (x, y coordinates)")
    print("  Hidden Layer 1: 64 neurons + ReLU")
    print("  Hidden Layer 2: 32 neurons + ReLU")
    print("  Hidden Layer 3: 16 neurons + ReLU")
    print("  Output Layer:    1 neuron + Sigmoid")
    print(
        "  Total parameters:",
        IntArray([p[].numels() for p in model.parameters()]).sum(),
    )
    print()

    # ============================================================================
    # Training Setup
    # ============================================================================

    var criterion = BCELoss[dtype]()
    var optimizer = SGD[dtype=dtype](
        model.parameters(), lr=learning_rate, momentum=momentum
    )

    print("Training Configuration:")
    print("  Loss function: Binary Cross-Entropy")
    print("  Optimizer: SGD with momentum")
    print()

    # ============================================================================
    # Training Loop
    # ============================================================================

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()

    var start_time = perf_counter_ns()

    for epoch in range(num_epochs):
        if epoch == 2500:
            curr_lr = optimizer.lr
            optimizer.set_lr(curr_lr / 10)
        if epoch == 3500:
            curr_lr = optimizer.lr
            optimizer.set_lr(curr_lr / 10)
        # ========================================================================
        # Training Phase
        # ========================================================================

        model.train()  # Enable gradient tracking
        criterion.train()  # Enable loss gradient computation

        var epoch_train_loss = 0.0
        var epoch_train_correct = 0
        var epoch_train_total = 0

        # Mini-batch training
        for train_batch in train_loader:
            # Forward pass
            var predictions = model(train_batch.features)
            var loss = criterion(predictions, train_batch.labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_train_loss += loss.item() * train_batch.batch_size

            var batch_correct_count, _ = accuracy(
                # var batch_correct_count, _ = binary_accuracy(
                predictions,
                train_batch.labels,
            )
            epoch_train_correct += batch_correct_count
            epoch_train_total += train_batch.batch_size

        # ========================================================================
        # Validation Phase
        # ========================================================================

        model.eval()  # Disable gradient tracking (no graph building)
        criterion.eval()  # Disable loss gradient computation

        var epoch_val_loss = 0.0
        var epoch_val_correct = 0
        var epoch_val_total = 0

        # Validation doesn't build computation graph (eval mode)
        for val_batch in val_loader:
            # Forward pass only (no backward, no graph)
            var val_predictions = model(val_batch.features)
            var val_loss = criterion(val_predictions, val_batch.labels)

            # Accumulate metrics
            epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct_count, _ = accuracy(
                # var val_correct_count, _ = binary_accuracy(
                val_predictions,
                val_batch.labels,
            )
            epoch_val_correct += val_correct_count
            epoch_val_total += val_batch.batch_size

        # ========================================================================
        # Logging and Monitoring
        # ========================================================================

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            # Calculate average metrics
            var avg_train_loss = epoch_train_loss / epoch_train_total
            var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
            var avg_val_loss = epoch_val_loss / epoch_val_total
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

            print(
                "Epoch",
                String(epoch).rjust(5),
                "| Train Loss:",
                String(avg_train_loss)[:7].rjust(7),
                "Acc:",
                String(train_accuracy)[:6].rjust(6),
                "%",
                "| Val Loss:",
                String(avg_val_loss)[:7].rjust(7),
                "Acc:",
                String(val_accuracy)[:6].rjust(6),
                "%",
            )

    var end_time = perf_counter_ns()
    var training_time_minutes = (end_time - start_time) / (1e9 * 60)

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("Total training time:", training_time_minutes, "minutes")
    print()

    # ============================================================================
    # Final Evaluation
    # ============================================================================

    print("=" * 80)
    print("Final Evaluation on Validation Set")
    print("=" * 80)
    print()

    model.eval()
    criterion.eval()

    var final_val_loss = 0.0
    var final_val_correct = 0
    var final_val_total = 0

    for val_batch in val_loader:
        var val_predictions = model(val_batch.features)
        var val_loss = criterion(val_predictions, val_batch.labels)

        final_val_loss += val_loss.item() * val_batch.batch_size

        var val_correct_count, _ = accuracy(val_predictions, val_batch.labels)
        # var val_correct_count, _ = binary_accuracy(val_predictions, val_batch.labels)
        final_val_correct += val_correct_count
        final_val_total += val_batch.batch_size

    var final_avg_val_loss = final_val_loss / final_val_total
    var final_val_accuracy = 100.0 * final_val_correct / final_val_total

    print("Final Validation Loss:", final_avg_val_loss)
    print("Final Validation Accuracy:", final_val_accuracy, "%")
    print()

    # ========================================================================
    # Performance Summary
    # ========================================================================

    var batches_per_epoch = len(train_loader)
    var total_batches = num_epochs * batches_per_epoch
    var ms_per_batch = (training_time_minutes * 60 * 1000) / total_batches

    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print("Total epochs:", num_epochs)
    print("Total batches processed:", total_batches)
    print("Average time per batch:", ms_per_batch, "ms")
    print("Average time per epoch:", ms_per_batch * batches_per_epoch, "ms")
    print()

    # Determine if training was successful
    if final_val_accuracy > 95.0 and final_avg_val_loss < 0.1:
        print("✓ Training successful! Model learned the spiral pattern.")
    elif final_val_accuracy > 90.0:
        print("✓ Training good! Model mostly learned the pattern.")
    elif final_val_accuracy > 80.0:
        print(
            "⚠ Training okay. Consider more epochs or different"
            " hyperparameters."
        )
    else:
        print(
            "✗ Training needs improvement. Try adjusting architecture or"
            " hyperparameters."
        )

    print("=" * 80)


fn main():
    """Entry point for the spiral classification example."""
    train_spiral_classifier()
