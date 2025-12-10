from tenmo import Tensor
from net import Linear, ReLU, Sequential, SGD
from crossentropy import CrossEntropyLoss
from testing import assert_true
from python import Python, PythonObject
from numpy_interop import from_ndarray, numpy_dtype
from data import *
from common_utils import now


fn test_cross_entropy_stability() raises:
    """Test if CrossEntropyLoss has numerical stability issues."""
    print("=" * 70)
    print("TESTING CROSS ENTROPY LOSS STABILITY")
    print("=" * 70 + "\n")

    alias dtype = DType.float32
    var criterion = CrossEntropyLoss[dtype]()
    criterion.train()

    print("Test 1: Normal logits (should be stable)")
    var normal_logits = Tensor[dtype].d2(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )
    var targets1 = Tensor[DType.int32].d1([2, 3])
    var loss1 = criterion(normal_logits, targets1)
    print("  Logits range: [0.1, 0.8]")
    print("  Loss:", loss1.item())
    print("  Status:", "OK" if loss1.item() < 10 else "FAIL")
    print()

    print("Test 2: Large positive logits (common after several epochs)")
    var large_pos_logits = Tensor[dtype].d2(
        [[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]]
    )
    var targets2 = Tensor[DType.int32].d1([2, 3])
    var loss2 = criterion(large_pos_logits, targets2)
    print("  Logits range: [10, 45]")
    print("  Loss:", loss2.item())
    print("  Status:", "OK" if loss2.item() < 100 else "FAIL - OVERFLOW")
    print()

    print("Test 3: Very large logits (can cause exp() overflow)")
    var huge_logits = Tensor[dtype].d2(
        [[50.0, 60.0, 70.0, 80.0], [55.0, 65.0, 75.0, 85.0]]
    )
    var targets3 = Tensor[DType.int32].d1([2, 3])
    var loss3 = criterion(huge_logits, targets3)
    print("  Logits range: [50, 85]")
    print("  Loss:", loss3.item())
    print("  Status:", "OK" if loss3.item() < 100 else "FAIL - SEVERE OVERFLOW")
    print()

    print("Test 4: Mixed positive and negative (should be stable)")
    var mixed_logits = Tensor[dtype].d2(
        [[-5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0]]
    )
    var targets4 = Tensor[DType.int32].d1([2, 3])
    var loss4 = criterion(mixed_logits, targets4)
    print("  Logits range: [-10, 10]")
    print("  Loss:", loss4.item())
    print("  Status:", "OK" if loss4.item() < 10 else "FAIL")
    print()

    print("Test 5: Extreme values (stress test)")
    var extreme_logits = Tensor[dtype].d2(
        [[100.0, 200.0, 300.0, 400.0], [150.0, 250.0, 350.0, 450.0]]
    )
    var targets5 = Tensor[DType.int32].d1([2, 3])
    var loss5 = criterion(extreme_logits, targets5)
    print("  Logits range: [100, 450]")
    print("  Loss:", loss5.item())
    var is_nan = loss5.item() != loss5.item()
    var is_inf = loss5.item() > 1e10
    if is_nan:
        print("  Status: FAIL - NaN detected!")
    elif is_inf:
        print("  Status: FAIL - Infinity/Overflow detected!")
    else:
        print("  Status: OK")
    print()

    print("=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    if loss2.item() > 100 or loss3.item() > 100:
        print("❌ CrossEntropyLoss has numerical stability issues!")
        print(
            "\nThe loss explodes with large logits. This is typically"
            " caused by:"
        )
        print("  1. Direct exp(logits) without the LogSumExp trick")
        print("  2. Missing max subtraction before exp()")
        print("  3. Overflow in softmax computation")
        print("\nRECOMMENDATION:")
        print(
            "  - Implement LogSumExp trick: max_val + log(sum(exp(x -"
            " max_val)))"
        )
        print("  - Add output clipping to your network")
        print("  - Use lower learning rates")
    else:
        print("✅ CrossEntropyLoss appears numerically stable")
        print("\nThe training explosion is likely due to:")
        print("  - Learning rate too high")
        print("  - Gradient explosion")
        print("  - Poor weight initialization")

    print("=" * 70)


fn test_training_logit_magnitudes() raises:
    """Check what logit magnitudes actually occur during training."""
    print("\n" + "=" * 70)
    print("TESTING ACTUAL TRAINING LOGIT MAGNITUDES")
    print("=" * 70 + "\n")

    # Quick training simulation
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")
    var np = Python.import_module("numpy")

    var train_data = loader.load()
    var train_images_np = train_data[0]
    var train_labels_np = train_data[1]

    alias feature_dtype = DType.float32
    alias label_dtype = DType.int32

    train_images_np = train_images_np.astype(numpy_dtype(feature_dtype))
    train_labels_np = train_labels_np.astype(numpy_dtype(label_dtype))

    var X_train = from_ndarray[feature_dtype](train_images_np, copy=True)
    var y_train = from_ndarray[label_dtype](train_labels_np, copy=True)
    X_train = X_train / 255.0

    var model = Sequential[feature_dtype]()
    model.append(
        Linear[feature_dtype](784, 128, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](128, 32, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](32, 10, xavier=True).into(),
    )

    var criterion = CrossEntropyLoss[feature_dtype]()
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    criterion.train()

    var batch_size = 64

    print("Training for a few batches and checking logit magnitudes...\n")

    for epoch in range(8):  # Go up to epoch 8 where explosion happened
        print("Epoch", epoch + 1, ":")

        # Train on first 5 batches
        for batch_idx in range(5):
            var start_idx = batch_idx * batch_size
            var end_idx = start_idx + batch_size

            var batch_X = X_train[start_idx:end_idx, :]
            var batch_y = y_train[start_idx:end_idx]

            var logits = model(batch_X)
            var loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check logit statistics on first and last batch
            if batch_idx == 0 or batch_idx == 4:
                var max_logit: Float32 = -1e10
                var min_logit: Float32 = 1e10
                var num_elements = logits.shape()[0] * logits.shape()[1]

                for i in range(logits.shape()[0]):
                    for j in range(logits.shape()[1]):
                        var val = logits[i, j]
                        if val > max_logit:
                            max_logit = val
                        if val < min_logit:
                            min_logit = val

                print(
                    "  Batch",
                    batch_idx,
                    "- Loss:",
                    loss.item(),
                    "| Logit range: [",
                    min_logit,
                    ",",
                    max_logit,
                    "]",
                )

        print()

    print("=" * 70)


fn main() raises:
    print("\n")
    test_cross_entropy_stability()
    test_training_logit_magnitudes()

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
