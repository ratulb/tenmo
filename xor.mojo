"""
XOR Problem Demonstration
=========================
Classic non-linearly separable problem requiring hidden layers.
"""

from tenmo import Tensor
from net import Sequential, Linear, Sigmoid, SGD, MSELoss
from time import perf_counter_ns
from math import sqrt

fn xor_classification() -> None:
    """Train a minimal neural network to solve XOR."""

    alias dtype = DType.float64

    # Configuration
    var num_epochs = 2000
    var learning_rate = 0.5
    var momentum = 0.9
    var log_interval = 200

    # XOR truth table
    var X = Tensor[dtype].d2([[0, 0], [0, 1], [1, 0], [1, 1]])
    var y = Tensor[dtype].d2([[0], [1], [1], [0]])

    # Model
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 4, xavier=True).into(),
        Sigmoid[dtype]().into(),
        Linear[dtype](4, 1, xavier=True).into(),
        Sigmoid[dtype]().into(),
    )

    # Training setup
    var criterion = MSELoss[dtype]()
    var optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    print("Training XOR solver...")
    var start_time = perf_counter_ns()

    model.train()
    criterion.train()

    for epoch in range(num_epochs):
        var pred = model(X)
        var loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            # Gradient norm
            var grad_norm_sq: Scalar[dtype] = 0.0
            for param in model.parameters():
                if param[].has_grad():
                    var g = param[].grad()
                    grad_norm_sq += (g * g).sum().item()

            var loss_str = String(loss.item())
            if len(loss_str) > 8:
                loss_str = loss_str[:8]

            var grad_str = String(sqrt(grad_norm_sq))
            if len(grad_str) > 8:
                grad_str = grad_str[:8]

            print("Epoch " + String(epoch).rjust(4) +
                  " | Loss: " + loss_str.rjust(8) +
                  " | Grad: " + grad_str.rjust(8))

            # Show predictions at key epochs
            if epoch in [0, 1000, num_epochs - 1]:
                model.eval()
                var final_pred = model(X)
                model.train()

                print("\nEpoch " + String(epoch) + " predictions:")
                for i in range(4):
                    var exp = Int(y[i, 0])
                    var pred_val = final_pred[i, 0]
                    var pred_str = String(pred_val)
                    if len(pred_str) > 6:
                        pred_str = pred_str[:6]
                    var err = abs(exp - pred_val)
                    var err_str = String(err)
                    if len(err_str) > 6:
                        err_str = err_str[:6]

                    print("  (" + String(Int(X[i,0])) + "," + String(Int(X[i,1])) +
                          ") → " + String(exp) + " | " + pred_str +
                          " (err: " + err_str + ")")
                print()

    var train_time = (perf_counter_ns() - start_time) / 1e9

    # Final evaluation
    model.eval()
    var final_pred = model(X)
    var final_loss = criterion(final_pred, y)

    var correct = 0
    var total_error = 0.0
    for i in range(4):
        var pred_class = 1 if final_pred[i, 0] > 0.5 else 0
        var true_class = Int(y[i, 0])
        if pred_class == true_class:
            correct += 1
        total_error += abs(final_pred[i, 0] - y[i, 0])

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print("Training time: " + String(train_time)[:6] + "s")
    print("Final loss: " + String(final_loss.item()))
    print("Accuracy: " + String(100.0 * correct / 4) + "%")
    print("Avg error: " + String(total_error / 4))
    print()

    # Success check
    if correct == 4:
        print("✓ Success: Network learned XOR perfectly")
    else:
        print("✗ Failed: Network did not learn XOR")

fn main():
    xor_classification()

