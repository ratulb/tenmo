from tensors import Tensor
from common_utils import panic
from utils import Variant


# --------------------
# Module struct
# --------------------
@fieldwise_init
struct Module[dtype: DType = DType.float32](Copyable & Movable):
    var layer: Layer[dtype]

    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        if self.layer.isa[Linear[dtype]]():
            return self.layer[Linear[dtype]](x)
        elif self.layer.isa[ReLU[dtype]]():
            return self.layer[ReLU[dtype]](x)
        else:
            panic("Unknown module type")
            return Tensor[dtype].scalar(
                0
            )  # Mojo needs a return but we would never reach here

    fn parameters(self) -> List[Tensor[dtype]]:
        if self.layer.isa[Linear[dtype]]():
            return self.layer[Linear[dtype]].parameters()
        elif self.layer.isa[ReLU[dtype]]():
            return self.layer[ReLU[dtype]].parameters()
        else:
            return List[Tensor[dtype]]()


# --------------------
# Linear
# --------------------
@fieldwise_init
struct Linear[dtype: DType = DType.float32](Copyable & Movable):
    var weights: Tensor[dtype]
    var bias: Tensor[dtype]

    fn __init__(out self, in_features: Int, out_features: Int):
        self.weights = Tensor[dtype].rand(
            [in_features, out_features], requires_grad=True
        )
        self.bias = Tensor[dtype].zeros([out_features], requires_grad=True)

    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        if x.shape[-1] != self.weights.shape[0]:
            panic("Linear forward: input dim mismatch")
        return x.matmul(self.weights) + self.bias

    fn parameters(self) -> List[Tensor[dtype]]:
        var p = List[Tensor[dtype]]()
        p.append(self.weights)
        p.append(self.bias)
        return p

    fn into(self) -> Module[dtype]:
        return Module[dtype](Layer[dtype](self))


# --------------------
# ReLU
# --------------------
@fieldwise_init
struct ReLU[dtype: DType = DType.float32](Copyable & Movable):
    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        # return x.relu()
        return Tensor[dtype].scalar(10)

    fn parameters(self) -> List[Tensor[dtype]]:
        return List[Tensor[dtype]]()

    fn into(self) -> Module[dtype]:
        return Module[dtype](Layer[dtype](self))


# --------------------
# Variant-based Sequential
# --------------------
alias Layer[dtype: DType] = Variant[Linear[dtype], ReLU[dtype]]


@fieldwise_init
struct Sequential[dtype: DType = DType.float32](Copyable & Movable):
    var modules: List[Module[dtype]]

    fn __init__(out self):
        self.modules = List[Module[dtype]]()

    fn append(mut self, m: Module[dtype]):
        self.modules.append(m)

    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        var out = x
        for m in self.modules:
            out = m(out)
        return out

    fn parameters(self) -> List[Tensor[dtype]]:
        var params = List[Tensor[dtype]]()

        for m in self.modules:
            for p in m.parameters():
                params.append(p)
        return params


# --------------------
# Example usage
# --------------------
fn main():
    # Build a network: Linear -> ReLU -> Linear
    var net = Sequential()  # Default DType is DType.float32
    net.append(Linear(4, 6).into())
    net.append(ReLU().into())
    net.append(Linear(6, 3).into())

    # Input
    var x = Tensor.rand([2, 4], requires_grad=True)

    # Forward pass
    var out = net(x)

    print("Input:")
    x.print()
    print("Output:")
    out.print()

    print("Number of learnable parameters: ", len(net.parameters()))
