# run_tensor.mojo  (sits at repo root, outside tenmo/)
from tenmo.tensor import Tensor

fn main() raises:
    # whatever quick test you want
    var t = Tensor[DType.float32].ones(3, 3)
    t.print()
