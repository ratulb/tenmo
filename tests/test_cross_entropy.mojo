from tensors import Tensor
from crossentropy import CrossEntropyLoss

fn main() raises:
    test_cross_entropy()
    print("passes")


fn test_cross_entropy() raises:
    print("Testing CrossEntropyLoss...")

    # Example 1: Basic classification
    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]],  # Sample 1  # Sample 2
        requires_grad=True,
    )

    var target = Tensor.d1([0, 1])  # Class indices

    # var criterion = CrossEntropyLoss(0, -1, Float32(0.1))
    var criterion = CrossEntropyLoss()
    var loss = criterion(logits, target)

    print("Loss:", loss.item())
    loss.backward()
    print("Gradient of logits:")
    logits.gradbox[].print()

    # Example 2: With ignore_index
    # var target_with_ignore = Tensor.d1([0, -1, 1])  # Ignore sample 2
    var target_with_ignore = Tensor.d1([0, -1])  # Ignore sample 2
    var loss_ignore = criterion(logits, target_with_ignore)
    print()
    print("Loss with ignore_index:", loss_ignore.item())

    loss_ignore.backward()

    print()

    print("Gradient of logits again:")
    logits.gradbox[].print()

    # Example 3: With ignore_index - 3 samples, one ignored
    var logits_3 = Tensor.d3(
        [[[2.0, 1.0], [0.5, 2.0], [1.0, 0.5]]]
    )  # 3 samples
    var target_ignore_3 = Tensor.d2(
        [[0, -1]]
    )  # Shape: (1, 2) - matches spatial dims
    var loss_ignore_3 = criterion(logits_3, target_ignore_3)
    print()
    print("Loss with target_ignore_3:", loss_ignore_3.item())

    loss_ignore_3.backward()

    print()

    print("Gradient of logits now:")
    logits.gradbox[].print()

    # Or for 3 samples with 2 spatial positions each:
    var logits_3_samples = Tensor.d3(
        [
            [[2.0, 1.0], [0.5, 2.0], [1.0, 0.5]],  # Sample 1
            [[1.0, 2.0], [0.5, 1.0], [2.0, 0.5]],  # Sample 2
            [[0.5, 1.0], [2.0, 0.5], [1.0, 2.0]],  # Sample 3
        ]
    )  # Shape: (3, 3, 2)

    var target_ignore_3_samples = Tensor.d2(
        [
            [0, -1],  # Sample 1: class 0 at pos 0, ignore at pos 1
            [-1, 1],  # Sample 2: ignore at pos 0, class 1 at pos 1
            [2, 0],  # Sample 3: class 2 at pos 0, class 0 at pos 1
        ]
    )  # Shape: (3, 2) - matches (N, spatial_dims)

    var loss_ignore_3_samples = criterion(
        logits_3_samples, target_ignore_3_samples
    )
    print()
    print("Loss with target_ignore_3_samples:", loss_ignore_3_samples.item())

    loss_ignore_3_samples.backward()

    print()

    print("Gradient of logits finally:")
    logits.gradbox[].print()
