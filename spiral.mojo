from tenmo import Tensor
from math import sin, cos, pi
from random import randn_float64
from common_utils import isnan
from math import sqrt
from time import perf_counter_ns
from data import *
from net import BCELoss

fn generate_spiral_data(
    n_points: Int = 100,
    n_rotations: Float64 = 3.0,
    noise: Float64 = 0.1
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

from net import Linear, Tanh, Sigmoid, Sequential, SGD, ReLU

struct SpiralNet[dtype: DType]:
    """Network designed for spiral classification."""

    # Architecture: 2 → 100 → 100 → 50 → 1
    var fc1: Linear[dtype]  # 2 → 100
    var fc2: Linear[dtype]  # 100 → 100
    var fc3: Linear[dtype]  # 100 → 50
    var fc4: Linear[dtype]  # 50 → 1

    fn __init__(out self):
        self.fc1 = Linear[dtype](2, 100)
        self.fc2 = Linear[dtype](100, 100)
        self.fc3 = Linear[dtype](100, 50)
        self.fc4 = Linear[dtype](50, 1)

    fn forward(mut self, x: Tensor[dtype]) -> Tensor[dtype]:
        var h1 = self.fc1(x).tanh()     # Tanh works well here
        var h2 = self.fc2(h1).tanh()
        var h3 = self.fc3(h2).tanh()
        var out = self.fc4(h3).sigmoid()
        return out


fn train_spiral_1():
    # Generate data
    var (X_train, y_train) = generate_spiral_data(
        #n_points=1000,
        n_points=500,
        n_rotations=3.0,
        noise=0.1
    )

    # Separate train/val
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    # Build model
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 100).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](100, 100).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](100, 50).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](50, 1).into(),
        Sigmoid[DType.float64]().into()
    )

    # Optimizer
    var optimizer = SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    # Training loop
    for epoch in range(10000):
        # Forward
        var pred = model(X_train)
        var loss = pred.binary_cross_entropy(y_train)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate every 1000 epochs
        if epoch % 500 == 0:
            var val_pred = model(X_val)
            var val_loss = val_pred.binary_cross_entropy(y_val)
            var accuracy = compute_accuracy(val_pred, y_val)

            print("Epoch", epoch)
            print("  Train Loss:", loss.item())
            print("  Val Loss:", val_loss.item())
            print("  Val Accuracy:", accuracy, "%")

fn train_spiral_2():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    # ✅ Simple 2-layer network with He init
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 128, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](128, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    # ✅ LR = 0.1 for ReLU
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5000):  # Might converge faster now
        var pred = model(X_train)
        var loss = pred.binary_cross_entropy(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var val_pred = model(X_val)
            var val_loss = val_pred.binary_cross_entropy(y_val)
            var accuracy = compute_accuracy(val_pred, y_val)

            print("Epoch", epoch)
            print("  Train Loss:", loss.item())
            print("  Val Loss:", val_loss.item())
            print("  Val Accuracy:", accuracy, "%")

        if epoch % 100 == 0:

            var weight_norm: Scalar[DType.float64] = 0
            var grad_norm: Scalar[DType.float64] = 0

            for param in model.parameters():
                weight_norm += (param[] * param[]).sum().item()
                if param[].has_grad():
                    var g = param[].grad()
                    grad_norm += (g * g).sum().item()

            print("Weight norm:", weight_norm)
            print("Gradient norm:", grad_norm)
            print("Prediction range: [", pred.min().item(), ",", pred.max().item(), "]")


from math import exp

fn compute_accuracy_10[dtype: DType, //](logits: Tensor[dtype], target: Tensor[dtype]) -> Float64:
    var correct = 0
    var total = logits.shape()[0]

    for i in range(total):
        # Sigmoid: 1 / (1 + exp(-x))
        var prob = 1.0 / (1.0 + exp(-logits[i, 0]))
        var predicted_class = 1 if prob > 0.5 else 0
        var true_class = Int(target[i, 0])
        if predicted_class == true_class:
            correct += 1

    return Float64(correct) / Float64(total) * 100.0

fn train_spiral_parked():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    # ✅ Simple 2-layer network with He init
    var model = Sequential[DType.float64]()
    _="""model.append(
        Linear[DType.float64](2, 128, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](128, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 1, xavier=False).into(),
        #Sigmoid[DType.float64]().into()
        )"""
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 32, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](32, 1, xavier=False).into()
    )

    # ✅ LR = 0.1 for ReLU
    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    #for epoch in range(5000):  # Might converge faster now
    var base_lr = 0.5
    for epoch in range(10000):
        # Reduce LR over time
        var current_lr = base_lr
        if epoch > 2000:
            current_lr = 0.3
        if epoch > 5000:
            current_lr = 0.1
        if epoch > 7000:
            current_lr = 0.01

        # Update optimizer LR (if your SGD supports it)
        optimizer.set_lr(current_lr)

        var pred = model(X_train)
        var loss = pred.binary_cross_entropy_with_logits(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var val_pred = model(X_val)
            var val_loss = val_pred.binary_cross_entropy_with_logits(y_val)
            var accuracy = compute_accuracy(val_pred, y_val)

            print("Epoch", epoch)
            print("  Train Loss:", loss.item())
            print("  Val Loss:", val_loss.item())
            print("  Val Accuracy:", accuracy, "%")

        if epoch % 100 == 0:

            var weight_norm: Scalar[DType.float64] = 0
            var grad_norm: Scalar[DType.float64] = 0

            for param in model.parameters():
                weight_norm += (param[] * param[]).sum().item()
                if param[].has_grad():
                    var g = param[].grad()
                    grad_norm += (g * g).sum().item()

            print("Weight norm:", weight_norm)
            print("Gradient norm:", grad_norm)
            print("Prediction range: [", pred.min().item(), ",", pred.max().item(), "]")

fn train_spiral():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 32, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](32, 1, xavier=False).into()
    )

    var optimizer = SGD(model.parameters(), lr=0.3, momentum=0.9)

    for epoch in range(5000):
        var pred = model(X_train)
        var loss = pred.binary_cross_entropy_with_logits(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var val_pred = model(X_val)
            var val_loss = val_pred.binary_cross_entropy_with_logits(y_val)
            var accuracy = compute_accuracy(val_pred, y_val)

            print("Epoch", epoch, "| Train:", loss.item(), "| Val:", val_loss.item(), "| Acc:", accuracy, "%")


#```

### Expected Training Curve
#```
#Epoch 0:     Loss: 0.25, Accuracy: 50%  (random)
#Epoch 2000:  Loss: 0.20, Accuracy: 65%  (learning basic patterns)
#Epoch 5000:  Loss: 0.12, Accuracy: 80%  (spirals emerging)
#Epoch 10000: Loss: 0.05, Accuracy: 92%  (good separation)
#Epoch 20000: Loss: 0.02, Accuracy: 96%+ (nearly perfect)

fn main_parked():
    _="""pair = generate_spiral_data()
    print()
    pair[0].print()
    print()
    pair[1].print()"""
    train_spiral()



fn test_tiny_overfit():
    """If this fails, there's a bug in forward/backward."""
    print("=== Testing Tiny Dataset Overfitting ===")

    # Just 4 points - 2 per class
    var X = Tensor[DType.float64].zeros(4, 2)
    X[0, 0] = 0.0; X[0, 1] = 0.0  # Class 0
    X[1, 0] = 0.1; X[1, 1] = 0.1  # Class 0
    X[2, 0] = 1.0; X[2, 1] = 1.0  # Class 1
    X[3, 0] = 0.9; X[3, 1] = 0.9  # Class 1

    var y = Tensor[DType.float64].zeros(4, 1)
    y[0, 0] = 0.0
    y[1, 0] = 0.0
    y[2, 0] = 1.0
    y[3, 0] = 1.0

    # Tiny network
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 10, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](10, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.0)

    print("Before training:")
    var pred_before = model(X)
    print("  Predictions:")
    pred_before.print()
    print("  Targets:")
    y.print()
    # Train for 1000 steps
    for epoch in range(1000):
        var pred = model(X)
        var loss = pred.binary_cross_entropy(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch", epoch, "Loss:", loss.item())

    print("\nAfter training:")
    var pred_after = model(X)
    print("  Predictions:")
    pred_after.print()
    print()
    print("  Targets:")
    y.print()

    # Check if it learned
    var accuracy = compute_accuracy(pred_after, y)
    print("  Accuracy:", accuracy, "%")

    if accuracy < 100.0:
        print("\n❌ FAILED: Should reach 100% on 4 points!")
        print("There's a bug in forward/backward passes.")
    else:
        print("\n✅ PASSED: Network can learn!")

fn test_gradients() -> Bool:
    """Check if gradients are computed and non-zero."""
    print("=== Testing Gradient Flow ===")

    var X = Tensor[DType.float64].zeros(2, 2)
    X[0, 0] = 1.0; X[0, 1] = 2.0
    X[1, 0] = 3.0; X[1, 1] = 4.0

    var y = Tensor[DType.float64].zeros(2, 1)
    y[0, 0] = 0.0
    y[1, 0] = 1.0

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 4, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](4, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var pred = model(X)
    var loss = pred.binary_cross_entropy(y)

    print("Loss:", loss.item())
    print("Pred:", pred)

    # Backward
    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.0)
    optimizer.zero_grad()
    loss.backward()

    # Check ALL parameters have gradients
    var params = model.parameters()
    var all_ok = True

    for i in range(len(params)):
        var param = params[i]
        print("\nParameter", i, ":")
        print("  Shape:", param[].shape())
        print("  Requires grad:", param[].requires_grad)

        if param[].has_grad():
            var grad = param[].grad()
            var grad_norm = abs(grad).sum().item()
            var grad_as_tensor = grad.copy().as_tensor()
            var grad_max = abs(grad_as_tensor).max().item()
            var grad_mean = grad_as_tensor.mean().item()

            print("  Grad norm:", grad_norm)
            print("  Grad max:", grad_max)
            print("  Grad mean:", grad_mean)

            if grad_norm == 0.0:
                print("  ❌ ZERO GRADIENT!")
                all_ok = False
            elif grad_norm < 1e-10:
                print("  ⚠️ Extremely small gradient (vanishing)")
                all_ok = False
            else:
                print("  ✅ Gradient OK")
        else:
            print("  ❌ NO GRADIENT COMPUTED!")
            all_ok = False

    if all_ok:
        print("\n✅ All gradients are flowing!")
    else:
        print("\n❌ Gradient problem detected!")

    return all_ok



fn test_tiny_overfit_detailed():
    print("=== Detailed Tiny Dataset Test ===")

    var X = Tensor[DType.float64].zeros(4, 2)
    X[0, 0] = 0.0; X[0, 1] = 0.0
    X[1, 0] = 0.1; X[1, 1] = 0.1
    X[2, 0] = 1.0; X[2, 1] = 1.0
    X[3, 0] = 0.9; X[3, 1] = 0.9

    var y = Tensor[DType.float64].zeros(4, 1)
    y[0, 0] = 0.0
    y[1, 0] = 0.0
    y[2, 0] = 1.0
    y[3, 0] = 1.0

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 10, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](10, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.0)

    # ✅ Check initial state
    print("=== Before Training ===")
    var pred_before = model(X)
    print("Predictions (before):")
    pred_before.print()  # ✅ Use .print() method
    print("\nTargets:")
    y.print()

    print("\nIndividual prediction values:")
    for i in range(4):
        print("  Sample", i, ":", pred_before[i, 0], "(target:", y[i, 0], ")")

    var params = model.parameters()
    var w0_initial = params[0][].buffer.data_buffer()[0]
    print("\nInitial weight[0,0]:", w0_initial)

    # ✅ Train with detailed logging
    print("\n=== Training ===")
    for epoch in range(1000):
        var pred = model(X)
        var loss = pred.binary_cross_entropy(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("\nEpoch", epoch)
            print("  Loss:", loss.item())
            print("  Predictions:")
            for i in range(4):
                var pred_val = pred[i, 0]
                var pred_class = 1 if pred_val > 0.5 else 0
                var true_class = Int(y[i, 0])
                var is_correct = pred_class == true_class
                var marker = "✓" if is_correct else "✗"
                print("    [", i, "]:", pred_val, "→ class", pred_class, "(", marker, ")")

    # ✅ Final check
    print("\n=== After Training ===")
    var pred_final = model(X)
    var loss_final = pred_final.binary_cross_entropy(y)

    print("Final loss:", loss_final.item())
    print("\nFinal predictions tensor:")
    pred_final.print()  # ✅ Use .print() method

    print("\nDetailed final predictions:")
    var correct = 0
    for i in range(4):
        var pred_val = pred_final[i, 0]
        var pred_class = 1 if pred_val > 0.5 else 0
        var true_class = Int(y[i, 0])
        var is_correct = pred_class == true_class

        if is_correct:
            correct += 1

        print("  Sample", i, ":")
        print("    Prediction value:", pred_val)
        print("    Predicted class:", pred_class)
        print("    True class:", true_class)
        print("    Correct:", is_correct)

    var accuracy = Float64(correct) / 4.0 * 100.0
    print("\nFinal Accuracy:", accuracy, "%")

    var w0_final = params[0][].buffer.data_buffer()[0]
    print("Weight[0,0] change:", w0_final - w0_initial)

    # ✅ Diagnosis
    print("\n=== Diagnosis ===")
    if loss_final.item() > 0.1:
        print("❌ Loss too high (", loss_final.item(), ") - not converging")
    elif accuracy < 100.0:
        print("❌ Accuracy not 100% - analyzing predictions...")

        # Check if predictions are stuck at 0.5
        var all_near_half = True
        for i in range(4):
            var pred_val = pred_final[i, 0]
            if abs(pred_val - 0.5) > 0.2:  # More than 0.2 away from 0.5
                all_near_half = False
                break

        if all_near_half:
            print("   → Predictions are all near 0.5 (stuck at boundary)")
            print("   → Possible causes:")
            print("      - Sigmoid output stuck at 0.5")
            print("      - Weights not updating properly")
            print("      - Gradients vanishing")
        else:
            print("   → Predictions are spread out but wrong")
            print("   → Possible causes:")
            print("      - Accuracy function bug")
            print("      - Wrong decision boundary")
    else:
        print("✅ Everything working! Network learned correctly.")



fn train_spiral_simple():
    print("=== Training Spiral with Simple Architecture ===")

    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    # ✅ Use EXACT same architecture that passed tiny test
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 10, xavier=False).into(),  # Same: 10 hidden units
        ReLU[DType.float64]().into(),
        Linear[DType.float64](10, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    # ✅ Same hyperparameters
    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.0)

    print("\nInitial predictions (first 5 samples):")
    var pred_init = model(X_val)
    for i in range(5):
        print("  [", i, "] pred:", pred_init[i, 0], "target:", y_val[i, 0])

    for epoch in range(5000):  # More epochs for harder problem
        var pred = model(X_train)
        var loss = pred.binary_cross_entropy(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var val_pred = model(X_val)
            var val_loss = val_pred.binary_cross_entropy(y_val)
            var accuracy = compute_accuracy(val_pred, y_val)

            print("\nEpoch", epoch)
            print("  Train Loss:", loss.item())
            print("  Val Loss:", val_loss.item())
            print("  Val Accuracy:", accuracy, "%")

            # ✅ Sample predictions
            print("  Sample predictions (first 5):")
            for i in range(5):
                var p = val_pred[i, 0]
                var t = y_val[i, 0]
                var pred_class = 1 if p > 0.5 else 0
                var true_class = Int(t)
                var ok = pred_class == true_class
                var marker = "✓" if ok else "✗"
                print("    [", i, "] pred:", p, "→", pred_class, "target:", true_class, marker)

    print("\n=== Final Results ===")
    var final_pred = model(X_val)
    var final_loss = final_pred.binary_cross_entropy(y_val)
    var final_acc = compute_accuracy(final_pred, y_val)

    print("Final Loss:", final_loss.item())
    print("Final Accuracy:", final_acc, "%")

    if final_acc > 85.0:
        print("✅ Network learned spiral pattern!")
    else:
        print("❌ Network failed to learn")


fn debug_spiral_data():
    print("=== Debugging Spiral Data Generation ===\n")

    var (X, y) = generate_spiral_data(n_points=100, n_rotations=3.0, noise=0.1)

    var total = X.shape()[0]
    print("Total samples:", total)
    print("Expected: 200 (100 per class)\n")

    # Check class distribution
    var class_0 = 0
    var class_1 = 0

    for i in range(total):
        if y[i, 0] < 0.5:
            class_0 += 1
        else:
            class_1 += 1

    print("Class distribution:")
    print("  Class 0:", class_0)
    print("  Class 1:", class_1)

    if class_0 != class_1:
        print("  ❌ IMBALANCED!")
    else:
        print("  ✅ Balanced")

    # Check if labels are in correct positions
    print("\nFirst 5 samples (should be class 0):")
    for i in range(5):
        print("  [", i, "] X:", X[i, 0], X[i, 1], "y:", y[i, 0])

    print("\nLast 5 samples (should be class 1):")
    for i in range(total - 5, total):
        print("  [", i, "] X:", X[i, 0], X[i, 1], "y:", y[i, 0])

    print("\nSamples around class boundary (index 99-104):")
    for i in range(99, min(104, total)):
        print("  [", i, "] X:", X[i, 0], X[i, 1], "y:", y[i, 0])

    # Check data ranges
    var x_min = X[0, 0]
    var x_max = X[0, 0]
    var y_coord_min = X[0, 1]
    var y_coord_max = X[0, 1]

    for i in range(total):
        if X[i, 0] < x_min: x_min = X[i, 0]
        if X[i, 0] > x_max: x_max = X[i, 0]
        if X[i, 1] < y_coord_min: y_coord_min = X[i, 1]
        if X[i, 1] > y_coord_max: y_coord_max = X[i, 1]

    print("\nInput data ranges:")
    print("  X[:,0]: [", x_min, ",", x_max, "]")
    print("  X[:,1]: [", y_coord_min, ",", y_coord_max, "]")

    # Check for NaN/Inf
    var has_nan = False
    for i in range(total):
        if isnan(X[i, 0]) or isnan(X[i, 1]) or isnan(y[i, 0]):
            has_nan = True
            print("  ❌ NaN at index", i)

    if not has_nan:
        print("  ✅ No NaN values")

    # Visualize spiral structure (text-based)
    print("\nSpiral structure check:")
    print("Class 0 samples - first 3 angles:")
    for i in range(3):
        print("  Sample", i, "position: (", X[i, 0], ",", X[i, 1], ")")

    print("Class 1 samples - first 3 angles:")
    for i in range(100, 103):
        print("  Sample", i, "position: (", X[i, 0], ",", X[i, 1], ")")

fn test_spiral_tiny():
    print("=== Testing on 20 Spiral Points ===\n")

    var (X_full, y_full) = generate_spiral_data(n_points=10, n_rotations=1.0, noise=0.0)

    # Just 20 points total (10 per class, 1 rotation, no noise)
    print("Dataset: 20 points, 10 per class, 1 rotation, no noise")
    print("Data:")
    X_full.print()
    print("Labels:")
    y_full.print()

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 10, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](10, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.1, momentum=0.0)

    print("\nTraining...")
    for epoch in range(3500):
        var pred = model(X_full)
        var loss = pred.binary_cross_entropy(y_full)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var acc = compute_accuracy(pred, y_full)
            print("\nEpoch", epoch)
            print("  Loss:", loss.item())
            print("  Accuracy:", acc, "%")

            if epoch % 1000 == 0:
                print("  Predictions:")
                for i in range(min(10, X_full.shape()[0])):
                    var p = pred[i, 0]
                    var t = y_full[i, 0]
                    var pred_class = 1 if p > 0.5 else 0
                    var true_class = Int(t)
                    print("    [", i, "] pred:", p, "→", pred_class, "target:", true_class)

    var final_pred = model(X_full)
    var final_acc = compute_accuracy(final_pred, y_full)

    print("\n=== Final Results ===")
    print("Accuracy:", final_acc, "%")

    if final_acc >= 95.0:
        print("✅ Can learn small spiral!")
    else:
        print("❌ Cannot even learn 20 points!")
        print("\nFinal predictions:")
        for i in range(X_full.shape()[0]):
            var p = final_pred[i, 0]
            var t = y_full[i, 0]
            print("  [", i, "] pred:", p, "target:", t)


fn test_spiral_xor_architecture():
    print("=== Spiral with XOR Architecture ===\n")

    # Small spiral dataset
    var (X, y) = generate_spiral_data(n_points=50, n_rotations=1.0, noise=0.05)

    print("Dataset: 100 points total")
    print("Architecture: Same as XOR (2→4→1)")

    # ✅ EXACT same architecture as XOR
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 4, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](4, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    # ✅ EXACT same hyperparameters as XOR
    var optimizer = SGD(model.parameters(), lr=0.5, momentum=0.0)

    for epoch in range(5000):
        var pred = model(X)
        var loss = pred.binary_cross_entropy(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            var acc = compute_accuracy(pred, y)
            print("Epoch", epoch, "Loss:", loss.item(), "Acc:", acc, "%")

            # Sample predictions
            print("  First 5 predictions:")
            for i in range(5):
                var p = pred[i, 0]
                var t = y[i, 0]
                var pred_class = 1 if p > 0.5 else 0
                var true_class = Int(t)
                var ok = pred_class == true_class
                print("    [", i, "] pred:", p, "→", pred_class, "target:", true_class, "✓" if ok else "✗")

    var final_acc = compute_accuracy(model(X), y)
    print("\nFinal Accuracy:", final_acc, "%")

    if final_acc >= 80.0:
        print("✅ Architecture can handle spiral!")
    else:
        print("❌ Same architecture fails on spiral")




fn normalize_data(X: Tensor[DType.float64]) -> Tensor[DType.float64]:
    """Normalize to mean=0, std=1."""
    var mean_x = 0.0
    var mean_y = 0.0
    var n = X.shape()[0]

    # Compute mean
    for i in range(n):
        mean_x += X[i, 0]
        mean_y += X[i, 1]
    mean_x /= Float64(n)
    mean_y /= Float64(n)

    # Compute std
    var std_x = 0.0
    var std_y = 0.0
    for i in range(n):
        std_x += (X[i, 0] - mean_x) ** 2
        std_y += (X[i, 1] - mean_y) ** 2
    std_x = sqrt(std_x / Float64(n))
    std_y = sqrt(std_y / Float64(n))

    print("Data stats:")
    print("  X mean:", mean_x, mean_y)
    print("  X std:", std_x, std_y)

    # Normalize
    var X_norm = Tensor[DType.float64].zeros(n, 2)
    for i in range(n):
        X_norm[i, 0] = (X[i, 0] - mean_x) / std_x
        X_norm[i, 1] = (X[i, 1] - mean_y) / std_y

    return X_norm^


fn train_spiral_normalized():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    # ✅ Normalize data
    var X_train_norm = normalize_data(X_train)
    var X_val_norm = normalize_data(X_val)

    var model = Sequential[DType.float64]()
    _="""model.append(
        Linear[DType.float64](2, 10, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](10, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )"""
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 16, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](16, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.075, momentum=0.95)

    for epoch in range(15000):
        var pred = model(X_train_norm)  # ✅ Use normalized
        var loss = pred.binary_cross_entropy(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            var val_pred = model(X_val_norm)  # ✅ Use normalized
            var val_acc = compute_accuracy(val_pred, y_val)
            print("Epoch", epoch, "Loss:", loss.item(), "Acc:", val_acc, "%")

            # Sample predictions
            print("  First 5 predictions:")
            for i in range(5):
                print("    pred:", val_pred[i, 0], "target:", y_val[i, 0])

fn train_spiral_fixed():
    var (X_train, y_train) = generate_spiral_data(400, 2.0, 0.01)
    var (X_val, y_val) = generate_spiral_data(200, 2.0, 0.01)

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 32, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](32, 16, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](16, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.05, momentum=0.1)

    for epoch in range(15000):
        if epoch == 10000:
            var current_lr = optimizer.lr
            optimizer.set_lr(current_lr/10)

        if epoch == 13000:
            var current_lr = optimizer.lr
            optimizer.set_lr(current_lr/10)

        if epoch == 14000:
            var current_lr = optimizer.lr
            optimizer.set_lr(current_lr/10)

        var pred = model(X_train)
        var loss = pred.binary_cross_entropy(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            var val_pred = model(X_val)
            var val_acc = compute_accuracy(val_pred, y_val)
            print("Epoch", epoch, "Loss:", loss.item(), "Acc:", val_acc, "%")

fn test_hardest_spiral():
    print("=== Testing Hardest Spiral ===")

    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.1)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.1)

    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 256, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](256, 128, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](128, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.075, momentum=0.025)

    for epoch in range(20000):
        #LR decay
        if epoch == 10000: optimizer.set_lr(0.01)
        if epoch == 15000: optimizer.set_lr(0.001)

        var pred = model(X_train)
        var loss = pred.binary_cross_entropy(y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            var val_pred = model(X_val)
            var val_acc = compute_accuracy(val_pred, y_val)
            print("Epoch", epoch, "Loss:", loss.item(), "Acc:", val_acc, "%")

fn train_with_batches_orig():
    var (X_train, y_train) = generate_spiral_data(500, 2.0, 0.01)
    var (X_val, y_val) = generate_spiral_data(250, 2.0, 0.01)
    alias dtype = DType.float64
    #✅ Use DataLoader with batching
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=32,
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=DType.float64](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    #Same model that worked
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 32, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](32, 16, xavier=False).into(),
        ReLU[DType.float64]().into(),
        Linear[DType.float64](16, 1, xavier=False).into(),
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD[dtype=dtype](model.parameters(), lr=0.05, momentum=0.1)

    #Train with batches
    for epoch in range(15000):
        var train_iter = train_loader.__iter__()
        var epoch_loss = 0.0
        var num_batches = 0

        for _batch in train_loader:


            while train_iter.__has_next__():
                var batch = train_iter.__next__()
                if batch.batch_size == 0: break

                var pred = model(batch.features)
                var loss = pred.binary_cross_entropy(batch.labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        if epoch % 1000 == 0:
            print("Epoch", epoch, "Avg Loss:", epoch_loss / num_batches)
            for batch in val_loader:

                var val_pred = model(batch.features)
                var val_acc = compute_accuracy(val_pred, batch.labels)
                print("Epoch", epoch, "Acc:", val_acc, "%")


fn train_with_batches_good_absent_validation_loss():
    var (X_train, y_train) = generate_spiral_data(500, 2.0, 0.01)
    var (X_val, y_val) = generate_spiral_data(250, 2.0, 0.01)
    alias dtype = DType.float64

    #✅ Use DataLoader with batching
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=32,
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=DType.float64](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    # Same model that worked - FIXED: Enable Xavier initialization
    var model = Sequential[DType.float64]()
    model.append(
        Linear[DType.float64](2, 64, xavier=False).into(),        # ✅ Disable Xavier
        ReLU[DType.float64]().into(),
        Linear[DType.float64](64, 32, xavier=False).into(),       # ✅ Disable Xavier
        ReLU[DType.float64]().into(),
        Linear[DType.float64](32, 16, xavier=False).into(),       # ✅ Disable Xavier
        ReLU[DType.float64]().into(),
        Linear[DType.float64](16, 1, xavier=False).into(),        # ✅ Disable Xavier
        Sigmoid[DType.float64]().into()
    )

    var optimizer = SGD[dtype=dtype](model.parameters(), lr=0.05, momentum=0.1)
    var num_epochs = 15000
    var start_training = perf_counter_ns()

    print("Starting training for epochs: ", num_epochs)

    # Train with batches
    for epoch in range(num_epochs):
        #var epoch_start = perf_counter_ns()
        model.train()

        var epoch_train_loss = 0.0
        var epoch_train_size = 0
        var epoch_correct = 0

        for train_batch in train_loader:

            #var batch_start = perf_counter_ns()
            var train_pred = model(train_batch.features)

            # ✅ FIXED: Use train_pred instead of undefined 'pred'
            var train_loss = train_pred.binary_cross_entropy(train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size  # ✅ Scale by batch size
            var batch_correct, _ = accuracy(train_pred, train_batch.labels)
            epoch_correct += batch_correct
            epoch_train_size += train_batch.batch_size

            #var batch_end = perf_counter_ns()
            #print("One batch of size:", train_batch.batch_size,  "took: ", (batch_end - batch_start) / 1e9, "secs")

        # ==================== Validation Phase ====================
        model.eval()
        #var epoch_val_loss = 0.0  # ✅ Added validation loss tracking
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in val_loader:
            var val_pred = model(val_batch.features)

            # ✅ Added validation loss calculation
            #var val_loss = val_pred.binary_cross_entropy(val_batch.labels)
            #epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct, _ = accuracy(val_pred, val_batch.labels)
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        if epoch % 1000 == 0:
            # Calculate training metrics for this epoch
            var avg_train_loss = epoch_train_loss / epoch_train_size  # ✅ Fixed variable name
            var train_accuracy = 100.0 * epoch_correct / epoch_train_size
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total
            #var avg_val_loss = epoch_val_loss / epoch_val_total  # ✅ Added avg validation loss

            print("Epoch", epoch,
                  "Train Loss:", avg_train_loss,
                  "Train Acc:", train_accuracy, "%",
                  #"Val Loss:", avg_val_loss,  # ✅ Report validation loss too
                  "Val Acc:", val_accuracy, "%")


        #var epoch_end = perf_counter_ns()
        #print("One epoch took: ", (epoch_end - epoch_start) / (1e9 * 60), "mins")

    var end_training = perf_counter_ns()
    print("Whole training loop took: ", (end_training - start_training) / (1e9 * 60), "mins")

fn accuracy[dtype: DType](pred: Tensor[dtype], target: Tensor[dtype]) -> Tuple[Int, Int]:
    var correct = 0
    var total = pred.shape()[0]

    for i in range(total):
        var predicted_class = 1 if pred[i, 0] > 0.5 else 0
        var true_class = Int(target[i, 0])
        if predicted_class == true_class:
            correct += 1
    return correct, total

fn compute_accuracy[dtype: DType, //](pred: Tensor[dtype], target: Tensor[dtype]) -> Float64:
    var correct = 0
    var total = pred.shape()[0]

    for i in range(total):
        var predicted_class = 1 if pred[i, 0] > 0.5 else 0
        var true_class = Int(target[i, 0])
        if predicted_class == true_class:
            correct += 1

    return Float64(correct) / Float64(total) * 100.0



fn train_with_batches():
    var (X_train, y_train) = generate_spiral_data(500, 2.0, 0.01)
    var (X_val, y_val) = generate_spiral_data(250, 2.0, 0.01)
    alias dtype = DType.float64

    # DataLoader with batching
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=32,
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=dtype](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    # Model
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 64, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](64, 32, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 16, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](16, 1, xavier=False).into(),
        Sigmoid[dtype]().into()
    )

    # Loss criterion with train/eval mode
    var criterion = BCELoss[dtype]()
    var optimizer = SGD[dtype=dtype](model.parameters(), lr=0.05, momentum=0.1)
    var num_epochs = 15000
    var start_training = perf_counter_ns()

    print("Starting training for epochs: ", num_epochs)

    for epoch in range(num_epochs):
        # ==================== Training Phase ====================
        model.train()
        criterion.train()  # Enable gradient tracking in loss

        var epoch_train_loss = 0.0
        var epoch_train_correct = 0
        var epoch_train_total = 0

        for train_batch in train_loader:
            var train_pred = model(train_batch.features)
            var train_loss = criterion(train_pred, train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size
            var batch_correct, _ = accuracy(train_pred, train_batch.labels)
            epoch_train_correct += batch_correct
            epoch_train_total += train_batch.batch_size

        # ==================== Validation Phase ====================
        model.eval()
        criterion.eval()  # Disable gradient tracking in loss

        var epoch_val_loss = 0.0
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in val_loader:
            var val_pred = model(val_batch.features)  # No graph built
            var val_loss = criterion(val_pred, val_batch.labels)  # No graph built

            epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct, _ = accuracy(val_pred, val_batch.labels)
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        # ==================== Reporting ====================
        if epoch % 1000 == 0:
            var avg_train_loss = epoch_train_loss / epoch_train_total
            var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
            var avg_val_loss = epoch_val_loss / epoch_val_total
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

            print("Epoch", epoch,
                  "| Train Loss:", avg_train_loss,
                  "Train Acc:", train_accuracy, "%",
                  "| Val Loss:", avg_val_loss,
                  "Val Acc:", val_accuracy, "%")

    var end_training = perf_counter_ns()
    print("\nTraining completed in:", (end_training - start_training) / (1e9 * 60), "mins")

    # Final evaluation
    model.eval()
    criterion.eval()
    var final_val_loss = 0.0
    var final_val_correct = 0
    var final_val_total = 0

    for val_batch in val_loader:
        var val_pred = model(val_batch.features)
        var val_loss = criterion(val_pred, val_batch.labels)
        final_val_loss += val_loss.item() * val_batch.batch_size
        var val_correct, _ = accuracy(val_pred, val_batch.labels)
        final_val_correct += val_correct
        final_val_total += val_batch.batch_size

    print("\n=== Final Results ===")
    print("Validation Loss:", final_val_loss / final_val_total)
    print("Validation Accuracy:", 100.0 * final_val_correct / final_val_total, "%")


fn train_deep_spiral():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.001)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.001)
    alias dtype = DType.float64

    # ✅ ADD BATCHING - Critical for deeper networks!
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=32,  # Smaller batches for harder problem
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=dtype](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    # DEEPER network (5 hidden layers)
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 128, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](128, 64, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](64, 32, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 16, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](16, 1, xavier=False).into(),
        Sigmoid[dtype]().into()
    )

    # Loss criterion with train/eval mode
    var criterion = BCELoss[dtype]()

    # Lower LR for deeper network
    var optimizer = SGD(model.parameters(), lr=0.035, momentum=0.085)
    var num_epochs = 5000
    var start_training = perf_counter_ns()

    print("Starting deep spiral training for epochs:", num_epochs)
    print("Problem difficulty: 3 rotations")

    for epoch in range(num_epochs):
        # LR decay schedule
        if epoch == 1500:
            optimizer.set_lr(0.02)
            print("  → Learning rate reduced to 0.02")
        if epoch == 2000:
            optimizer.set_lr(0.01)
            print("  → Learning rate reduced to 0.01")
        if epoch == 3000:
            optimizer.set_lr(0.001)
            print("  → Learning rate reduced to 0.001")

        # ==================== Training Phase ====================
        model.train()
        criterion.train()

        var epoch_train_loss = 0.0
        var epoch_train_correct = 0
        var epoch_train_total = 0

        for train_batch in train_loader:
            var train_pred = model(train_batch.features)
            var train_loss = criterion(train_pred, train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size
            var batch_correct, _ = accuracy(train_pred, train_batch.labels)
            epoch_train_correct += batch_correct
            epoch_train_total += train_batch.batch_size

        # ==================== Validation Phase ====================
        model.eval()
        criterion.eval()

        var epoch_val_loss = 0.0
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in val_loader:
            var val_pred = model(val_batch.features)  # No graph
            var val_loss = criterion(val_pred, val_batch.labels)  # No graph

            epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct, _ = accuracy(val_pred, val_batch.labels)
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        # ==================== Reporting ====================
        if epoch % 500 == 0:
            var avg_train_loss = epoch_train_loss / epoch_train_total
            var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
            var avg_val_loss = epoch_val_loss / epoch_val_total
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

            print("Epoch", epoch,
                  "| Train Loss:", avg_train_loss,
                  "Train Acc:", train_accuracy, "%",
                  "| Val Loss:", avg_val_loss,
                  "Val Acc:", val_accuracy, "%")

    var end_training = perf_counter_ns()
    print("\nTraining completed in:", (end_training - start_training) / (1e9 * 60), "mins")

    # Final evaluation
    model.eval()
    criterion.eval()
    var final_val_loss = 0.0
    var final_val_correct = 0
    var final_val_total = 0

    for val_batch in val_loader:
        var val_pred = model(val_batch.features)
        var val_loss = criterion(val_pred, val_batch.labels)
        final_val_loss += val_loss.item() * val_batch.batch_size
        var val_correct, _ = accuracy(val_pred, val_batch.labels)
        final_val_correct += val_correct
        final_val_total += val_batch.batch_size

    print("\n=== Final Results ===")
    print("Validation Loss:", final_val_loss / final_val_total)
    print("Validation Accuracy:", 100.0 * final_val_correct / final_val_total, "%")



fn train_deep_spiral_simple_arch_and_noise_increased():
    var (X_train, y_train) = generate_spiral_data(500, 3.0, 0.01)
    var (X_val, y_val) = generate_spiral_data(250, 3.0, 0.01)
    alias dtype = DType.float64

    # ✅ ADD BATCHING - Critical for deeper networks!
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=32,  # Smaller batches for harder problem
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=dtype](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    # DEEPER network (5 hidden layers)
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 32, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 16, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](16, 1, xavier=False).into(),
        Sigmoid[dtype]().into()
    )

    # Loss criterion with train/eval mode
    var criterion = BCELoss[dtype]()

    # Lower LR for deeper network
    var optimizer = SGD(model.parameters(), lr=0.035, momentum=0.085)
    var num_epochs = 5000
    var start_training = perf_counter_ns()

    print("Starting deep spiral training for epochs:", num_epochs)
    print("Problem difficulty: 3 rotations")

    for epoch in range(num_epochs):
        # LR decay schedule
        if epoch == 1500:
            optimizer.set_lr(0.02)
            print("  → Learning rate reduced to 0.02")
        if epoch == 2000:
            optimizer.set_lr(0.01)
            print("  → Learning rate reduced to 0.01")
        if epoch == 3000:
            optimizer.set_lr(0.001)
            print("  → Learning rate reduced to 0.001")

        # ==================== Training Phase ====================
        model.train()
        criterion.train()

        var epoch_train_loss = 0.0
        var epoch_train_correct = 0
        var epoch_train_total = 0

        for train_batch in train_loader:
            var train_pred = model(train_batch.features)
            var train_loss = criterion(train_pred, train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size
            var batch_correct, _ = accuracy(train_pred, train_batch.labels)
            epoch_train_correct += batch_correct
            epoch_train_total += train_batch.batch_size

        # ==================== Validation Phase ====================
        model.eval()
        criterion.eval()

        var epoch_val_loss = 0.0
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in val_loader:
            var val_pred = model(val_batch.features)  # No graph
            var val_loss = criterion(val_pred, val_batch.labels)  # No graph

            epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct, _ = accuracy(val_pred, val_batch.labels)
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        # ==================== Reporting ====================
        if epoch % 500 == 0:
            var avg_train_loss = epoch_train_loss / epoch_train_total
            var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
            var avg_val_loss = epoch_val_loss / epoch_val_total
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

            print("Epoch", epoch,
                  "| Train Loss:", avg_train_loss,
                  "Train Acc:", train_accuracy, "%",
                  "| Val Loss:", avg_val_loss,
                  "Val Acc:", val_accuracy, "%")

    var end_training = perf_counter_ns()
    print("\nTraining completed in:", (end_training - start_training) / (1e9 * 60), "mins")

    # Final evaluation
    model.eval()
    criterion.eval()
    var final_val_loss = 0.0
    var final_val_correct = 0
    var final_val_total = 0

    for val_batch in val_loader:
        var val_pred = model(val_batch.features)
        var val_loss = criterion(val_pred, val_batch.labels)
        final_val_loss += val_loss.item() * val_batch.batch_size
        var val_correct, _ = accuracy(val_pred, val_batch.labels)
        final_val_correct += val_correct
        final_val_total += val_batch.batch_size

    print("\n=== Final Results ===")
    print("Validation Loss:", final_val_loss / final_val_total)
    print("Validation Accuracy:", 100.0 * final_val_correct / final_val_total, "%")


fn train_deep_spiral_simple_arch_and_noise_increased_2():
    var (X_train, y_train) = generate_spiral_data(1000, 5.0, 0.0)
    var (X_val, y_val) = generate_spiral_data(500, 5.0, 0.0)
    alias dtype = DType.float64

    # ✅ ADD BATCHING - Critical for deeper networks!
    var train_dataset = TensorDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype=dtype](
        train_dataset^,
        batch_size=64,  # Smaller batches for harder problem
        reshuffle=True,
        drop_last=False
    )

    var val_dataset = TensorDataset(X_val, y_val)
    var val_loader = DataLoader[dtype=dtype](
        val_dataset^,
        batch_size=64,
        reshuffle=False
    )

    # DEEPER network (5 hidden layers)
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 32, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 16, xavier=False).into(),
        ReLU[dtype]().into(),
        Linear[dtype](16, 1, xavier=False).into(),
        Sigmoid[dtype]().into()
    )

    # Loss criterion with train/eval mode
    var criterion = BCELoss[dtype]()

    # Lower LR for deeper network
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    var num_epochs = 3000
    var start_training = perf_counter_ns()

    print("Starting deep spiral training for epochs:", num_epochs)
    print("Problem difficulty: 5 rotations, noise: 0.0")

    for epoch in range(num_epochs):
        # LR decay schedule
        _="""if epoch == 1500:
            optimizer.set_lr(0.025)
            print("  → Learning rate reduced to 0.025")"""
        if epoch == 2000:
            optimizer.set_lr(0.001)
            print("  → Learning rate reduced to 0.001")
        _="""if epoch == 3000:
            optimizer.set_lr(0.001)
            print("  → Learning rate reduced to 0.001")"""

        # ==================== Training Phase ====================
        model.train()
        criterion.train()

        var epoch_train_loss = 0.0
        var epoch_train_correct = 0
        var epoch_train_total = 0

        for train_batch in train_loader:
            var train_pred = model(train_batch.features)
            var train_loss = criterion(train_pred, train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size
            var batch_correct, _ = accuracy(train_pred, train_batch.labels)
            epoch_train_correct += batch_correct
            epoch_train_total += train_batch.batch_size

        # ==================== Validation Phase ====================
        model.eval()
        criterion.eval()

        var epoch_val_loss = 0.0
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in val_loader:
            var val_pred = model(val_batch.features)  # No graph
            var val_loss = criterion(val_pred, val_batch.labels)  # No graph

            epoch_val_loss += val_loss.item() * val_batch.batch_size

            var val_correct, _ = accuracy(val_pred, val_batch.labels)
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        # ==================== Reporting ====================
        if epoch % 500 == 0:
            var avg_train_loss = epoch_train_loss / epoch_train_total
            var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
            var avg_val_loss = epoch_val_loss / epoch_val_total
            var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

            print("Epoch", epoch,
                  "| Train Loss:", avg_train_loss,
                  "Train Acc:", train_accuracy, "%",
                  "| Val Loss:", avg_val_loss,
                  "Val Acc:", val_accuracy, "%")

    var end_training = perf_counter_ns()
    print("\nTraining completed in:", (end_training - start_training) / (1e9 * 60), "mins")

    # Final evaluation
    model.eval()
    criterion.eval()
    var final_val_loss = 0.0
    var final_val_correct = 0
    var final_val_total = 0

    for val_batch in val_loader:
        var val_pred = model(val_batch.features)
        var val_loss = criterion(val_pred, val_batch.labels)
        final_val_loss += val_loss.item() * val_batch.batch_size
        var val_correct, _ = accuracy(val_pred, val_batch.labels)
        final_val_correct += val_correct
        final_val_total += val_batch.batch_size

    print("\n=== Final Results ===")
    print("Validation Loss:", final_val_loss / final_val_total)
    print("Validation Accuracy:", 100.0 * final_val_correct / final_val_total, "%")




fn main():
    #test_tiny_overfit()
    #test_gradients()
    #test_tiny_overfit_detailed()
    #train_spiral_simple()
    #debug_spiral_data()#Good
    #test_spiral_tiny()# Good
    #test_spiral_xor_architecture()#Good

    #train_spiral_fixed() #too Good
    #test_hardest_spiral() # Not yet good
    #train_with_batches()# Working
    #train_deep_spiral() #This good too
    #train_with_batches()
    #train_spiral_normalized() #No good as it is
    #train_deep_spiral_simple_arch_and_noise_increased()#very good
    train_deep_spiral_simple_arch_and_noise_increased_2()

