from tensors import Tensor
from shapes import Shape
from intlist import IntList
from layers import Linear, ReLU, Sequential
from crossentropy import CrossEntropyLoss
from sgd import SGD
from testing import assert_true
from utils.numerics import max_finite
from common_utils import log_debug, CYAN, RED


fn make_synthetic_dataset(
    n_samples: Int,
) -> Tuple[Tensor[DType.float32], Tensor[DType.float32]]:
    # Images: n_samples × 1 × 8 × 8  (simple 8×8 grayscale blobs)
    var X = Tensor[DType.float32].zeros(Shape([n_samples, 1, 8, 8]))
    var y = Tensor[DType.float32].zeros(Shape(n_samples))

    for i in range(n_samples):
        # draw a random label 0–1
        var label = Tensor.randint32([1], 0, 2).float()
        y[i] = label[0]

        # simple pattern generator
        if label[0] == 0:
            # horizontal bar
            for r in range(3, 5):
                for c in range(8):
                    X[IntList([i, 0, r, c])] = 1.0
        else:
            # vertical bar
            for r in range(8):
                for c in range(3, 5):
                    X[IntList([i, 0, r, c])] = 1.0

    # flatten to (n_samples × 64)
    var X_flat = X.flatten[False](
        start_dim=1
    )  # squash everything but batch dim
    return (X_flat, y)


fn test_tiny_training_smoke() raises:
    print("test_tiny_training_smoke_and_flatten_grad")

    # --- synthetic training ---
    var (X, y) = make_synthetic_dataset(32)
    # var X_flat = X.flatten(1)  # flatten from dim=1
    # print(X.shape, y.shape, X_flat.shape)
    var model = Sequential(
        [
            Linear(8 * 8, 16).into(),
            ReLU().into(),
            Linear(16, 10).into(),
        ]
    )
    # model.print_summary([1, 64])
    var optim = SGD(model.parameters_ptrs(), 0.01, False)
    var criterion = CrossEntropyLoss()

    for epoch in range(10):
        var logits = model(X)
        var loss = criterion(logits, y)
        log_debug(
            "epoch: " + epoch.__str__() + ", loss: " + loss.item().__str__()
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

    # --- flatten gradient correctness (contiguous) ---
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var f = a.flatten()
    var loss2 = f.sum()
    loss2.backward()

    var expected = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.gradbox[].all_close(expected))

    # --- flatten gradient correctness (view / strided) ---
    var b = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var v = b.slice(1, 3, 1, 1)  # take columns 1:3 → [[2,3],[5,6]]
    var f2 = v.flatten()
    var loss3 = f2.sum()
    loss3.backward(42)

    var expected2 = Tensor.d2([[0.0, 42.0, 42.0], [0.0, 42.0, 42.0]])
    assert_true(b.gradbox[].all_close(expected2))


fn test_tiny_training_deterministic_smoke() raises:
    print("test_tiny_training_deterministic_smoke")

    # synthetic dataset
    var X, y = make_synthetic_dataset(32)  # small fixed size

    # simple MLP
    var model = Sequential(
        [
            Linear(
                X.shape[1], 16, init_seed=42
            ).into(),  # fix seed for reproducibility
            ReLU().into(),
            Linear(16, 2, init_seed=42).into(),
        ]
    )

    # optimizer and loss
    var optim = SGD(
        params=model.parameters_ptrs(), lr=0.0855, zero_grad_post_step=True
    )
    var criterion = CrossEntropyLoss()
    var loss: Tensor[DType.float32] = Tensor.scalar(max_finite[DType.float32]())

    for epoch in range(30):
        # optim.zero_grad()
        var logits = model(X)
        loss = criterion(logits, y)
        log_debug(
            "epoch: " + epoch.__str__() + ", loss: " + loss.item().__str__()
        )

        loss.backward()
        optim.step()

    # assert deterministic final loss
    diff = abs(loss.item() - 0.000027)
    tolerance = Float32(1e-4)
    loss_dev_msg = "Final loss deviation: {0} (tolerance: {1})".format(
        diff, tolerance
    )
    print(loss_dev_msg)
    try:
        assert_true(diff < tolerance)
        print(CYAN, "Deterministic final loss within tolerance ✅ ")
    except e:
        err_msg = (
            "Final loss exceeds tolerance limit ({0}) ❌  (diff: {1})".format(
                tolerance, diff
            )
        )
        print(RED, err_msg)


fn main() raises:
    test_tiny_training_smoke()
    test_tiny_training_deterministic_smoke()
