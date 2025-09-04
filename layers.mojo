from tensors import Tensor
from common_utils import panic, log_debug, RED, CYAN, MAGENTA
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

    fn num_parameters(self) -> Int:
        if self.layer.isa[Linear[dtype]]():
            return self.layer[Linear[dtype]].num_parameters()
        elif self.layer.isa[ReLU[dtype]]():
            return self.layer[ReLU[dtype]].num_parameters()
        else:
            return 0

    # Free all learnable params inside this module
    fn free_params(self):
        if self.layer.isa[Linear[dtype]]():
            var lin = self.layer[Linear[dtype]]
            log_debug(
                "Freeing Linear weights - shape: "
                + lin.weights.shape.__str__(),
                RED,
            )
            lin.weights.free()
            log_debug(
                "Freeing Linear bias - shape: " + lin.bias.shape.__str__(), RED
            )
            lin.bias.free()
        elif self.layer.isa[ReLU[dtype]]():
            log_debug("ReLU has no learnable params", MAGENTA)

    fn free_all(self):
        # Currently just frees parameters; if layer had internal buffers, free them here
        self.free_params()


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

    fn num_parameters(self) -> Int:
        return self.weights.numels() + self.bias.numels()


#


# --------------------
# ReLU
# --------------------
@fieldwise_init
struct ReLU[dtype: DType = DType.float32](Copyable & Movable):
    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        return x.relu()

    fn parameters(self) -> List[Tensor[dtype]]:
        return List[Tensor[dtype]]()

    fn into(self) -> Module[dtype]:
        return Module[dtype](Layer[dtype](self))

    fn num_parameters(self) -> Int:
        return 0


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

    fn num_parameters(self) -> Int:
        var total: Int = 0
        for p in self.parameters():
            total += p.numels()  # each tensor knows its total elements
        return total

    # Free just parameters across all modules
    fn free_params(self):
        for m in self.modules:
            m.free_params()

    # Free everything (params + clear module list)
    fn free_all(mut self):
        log_debug("Freeing Sequential modules...", CYAN)
        for m in self.modules:
            m.free_all()
        self.modules.clear()
        log_debug("Sequential cleared", CYAN)

    fn print_summary(self):
        print("Sequential Model Summary")
        print("------------------------")
        for i in range(len(self.modules)):
            m = self.modules[i]
            var name = "Unknown"
            var nparams = 0
            if m.layer.isa[Linear[dtype]]():
                name = "Linear"
                nparams = m.layer[Linear[dtype]].num_parameters()
            elif m.layer.isa[ReLU[dtype]]():
                name = "ReLU"
                nparams = m.layer[ReLU[dtype]].num_parameters()

            print("Layer ", i, ": ", name, " | Parameters: ", nparams)
        print("------------------------")
        print("Total learnable parameters: ", self.num_parameters())

    fn print_summary(self, input_shape: List[Int]):
        print("Sequential Model Summary")
        print("------------------------")

        var x = Tensor[dtype].zeros(input_shape, requires_grad=False)

        for i in range(len(self.modules)):
            m = self.modules[i]
            var name = "Unknown"
            var nparams = m.num_parameters()
            var in_shape = x.shape
            var details = ""
            var trainable = "No"

            if m.layer.isa[Linear[dtype]]():
                name = "Linear"
                l = m.layer[Linear[dtype]]
                details = (
                    "weight="
                    + l.weights.shape.__str__()
                    + ", bias="
                    + l.bias.shape.__str__()
                )
                # if either weight or bias requires grad, mark trainable
                if l.weights.requires_grad or l.bias.requires_grad:
                    trainable = "Yes"
                x = l(x)

            elif m.layer.isa[ReLU[dtype]]():
                name = "ReLU"
                details = ""
                trainable = "No"
                x = m.layer[ReLU[dtype]](x)

            var out_shape = x.shape

            print(
                "Layer ",
                i,
                ": ",
                name,
                " | Input: ",
                in_shape,
                " -> Output: ",
                out_shape,
                " | Parameters: ",
                nparams,
                (" | " + details if details != "" else ""),
                " | Trainable: ",
                trainable,
            )

        print("------------------------")
        print("Total learnable parameters: ", self.num_parameters())


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

    print("Number of learnable parameters: \n", len(net.parameters()), "\n")
    print("Out has backward fn? \n", out.has_backward_fn())
    out.backward()
    print("Total number of learnable scalars: ", net.num_parameters())
    print()
    print("gradbox\n")

    x.gradbox[].print()
    print()

    for p in net.parameters():
        print("Param grad:\n")
        p.gradbox[].print()

    net.print_summary([2, 4])
    net.free_all()

