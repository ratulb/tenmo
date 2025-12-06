from tenmo import Tensor
from shapes import Shape
from intarray import IntArray
from layers import Linear, ReLU, Sequential
from crossentropy import CrossEntropyLoss
from sgd import SGD
from testing import assert_true
from utils.numerics import max_finite
from common_utils import log_debug


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
                    X[IntArray([i, 0, r, c])] = 1.0
        else:
            # vertical bar
            for r in range(8):
                for c in range(3, 5):
                    X[IntArray([i, 0, r, c])] = 1.0

    # flatten to (n_samples × 64)
    var X_flat = X.flatten[track_grad=False](
        start_dim=1
    )  # squash everything but batch dim
    return (X_flat, y)


fn test_tiny_training_smoke() raises:
    print("test_tiny_training_smoke")

    var (X, y) = make_synthetic_dataset(5)  # small dataset

    var model = Sequential[DType.float32](
        [
            Linear[DType.float32](64, 32).into(),
            ReLU[DType.float32]().into(),
            Linear[DType.float32](32, 2).into(),
        ]
    )

    # var criterion = CrossEntropyLoss()
    var criterion = CrossEntropyLoss[DType.float32]()
    var optim = SGD[DType.float32](
        params=model.parameters_ptrs(),
        lr=0.1,
        zero_grad_post_step=False
        # params=model.parameters_ptrs(), lr=0.1
    )

    for epoch in range(10):
        optim.zero_grad()
        # forward
        var logits = model(X)
        var loss = criterion(logits, y)

        loss.backward()
        optim.step()

        log_debug(
            "epoch: " + epoch.__str__() + ", loss: " + loss.item().__str__()
        )


fn test_tiny_training_smoke_and_flatten_grad() raises:
    print("test_tiny_training_smoke_and_flatten_grad")

    # --- synthetic training ---
    var (X, y) = make_synthetic_dataset(32)
    var X_flat = X.flatten(1)  # flatten from dim=1
    var model = Sequential(
        [
            Linear(8 * 8, 16).into(),
            ReLU().into(),
            Linear(16, 10).into(),
        ]
    )
    model.print_summary([32, 64])
    var optim = SGD(model.parameters_ptrs(), 0.1, False)
    var criterion = CrossEntropyLoss()

    for epoch in range(15):
        var logits = model(X_flat)
        var loss = criterion(logits, y)
        log_debug(
            "epoch: " + epoch.__str__() + ", loss: " + loss.item().__str__()
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

    # --- flatten gradient correctness (contiguous) ---
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    var f = a.flatten()
    var loss2 = f.sum()
    loss2.backward()

    var expected = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).float()
    assert_true(a.gradbox[].all_close(expected))

    # --- flatten gradient correctness (view / strided) ---
    var b = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    var v = b.slice(1, 3, 1, 1)  # take columns 1:3 → [[2,3],[5,6]]
    var f2 = v.flatten()
    var loss3 = f2.sum()
    loss3.backward(42)

    var expected2 = Tensor.d2([[0.0, 42.0, 42.0], [0.0, 42.0, 42.0]]).float()
    assert_true(b.gradbox[].all_close(expected2))


fn test_tiny_training_deterministic_smoke() raises:
    print("test_tiny_training_deterministic_smoke")

    # fix seed for reproducibility
    # seed(12345)

    # synthetic dataset
    var X, y = make_synthetic_dataset(32)  # small fixed size
    var X_flat = X.flatten(1)  # match input shape

    # simple MLP
    var model = Sequential(
        [
            Linear(X_flat.shape()[1], 16).into(),
            ReLU().into(),
            Linear(16, 2, init_seed=42).into(),
        ]
    )

    # optimizer and loss
    var optim = SGD(
        params=model.parameters_ptrs(), lr=0.1, zero_grad_post_step=True
    )
    var criterion = CrossEntropyLoss()
    var loss: Tensor[DType.float32] = Tensor.scalar(max_finite[DType.float32]())

    for epoch in range(10):
        # optim.zero_grad()
        var logits = model(X_flat)
        loss = criterion(logits, y)
        log_debug(
            "epoch: " + epoch.__str__() + ", loss: " + loss.item().__str__()
        )

        loss.backward()
        optim.step()

    # assert deterministic final loss
    log_debug("diff: " + abs(loss.item() - 0.000027).__str__())
    assert_true(abs(loss.item() - 0.000027) < 1e-3)


fn main() raises:
    # test_tiny_training_smoke()
    # test_tiny_training_deterministic_smoke()
    test_tiny_training_smoke_and_flatten_grad()
