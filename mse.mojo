from tensors import Tensor
from intlist import IntList


# -----------------------------------------
# Mean Squared Error Loss
# -----------------------------------------
struct MSELoss[dtype: DType = DType.float32]:
    fn __init__(out self):
        pass

    fn __call__(
        self, preds: Tensor[dtype], target: Tensor[dtype]
    ) -> Tensor[dtype]:
        # (1/N) * Σ (input - target)^2
        diff = preds - target
        #loss = (diff * diff).mean(IntList(), False, True)
        loss = (diff * diff).mean()
        return loss


fn main():
    print("passes")
