# layers.mojo
from tensors import Tensor
from common_utils import panic, log_debug, RED, CYAN, MAGENTA, addr
from utils import Variant

# --------------------
# Module / Layer / Sequential definitions (safe pointers)
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
            return Tensor[dtype].scalar(0)

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
        self.free_params()

    fn parameters_ptrs(self) -> List[UnsafePointer[Tensor[dtype]]]:
        var ptrs = List[UnsafePointer[Tensor[dtype]]]()
        if self.layer.isa[Linear[dtype]]():
            var l = self.layer[Linear[dtype]]
            ptrs.append(addr(l.weights))
            ptrs.append(addr(l.bias))
        return ptrs


@fieldwise_init
struct Linear[dtype: DType = DType.float32](Copyable & Movable):
    var weights: Tensor[dtype]
    var bias: Tensor[dtype]

    fn __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
    ):
        self.weights = Tensor[dtype].rand(
            [in_features, out_features], init_seed=init_seed, requires_grad=True
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

    fn parameters_ptrs(self) -> List[UnsafePointer[Tensor[dtype]]]:
        var ptrs = List[UnsafePointer[Tensor[dtype]]]()
        ptrs.append(addr(self.weights))
        ptrs.append(addr(self.bias))
        return ptrs


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

    fn parameters_ptrs(self) -> List[UnsafePointer[Tensor[dtype]]]:
        return List[UnsafePointer[Tensor[dtype]]]()


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
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            out = m(out)
        return out

    fn parameters(self) -> List[Tensor[dtype]]:
        var params = List[Tensor[dtype]]()
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            for p in m.parameters():
                params.append(p)
        return params

    fn num_parameters(self) -> Int:
        var total: Int = 0
        for p in self.parameters():
            total += p.numels()
        return total

    fn free_params(self):
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            m.free_params()

    fn free_all(mut self):
        log_debug("Freeing Sequential modules...", CYAN)
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            m.free_all()
        self.modules.clear()
        log_debug("Sequential cleared", CYAN)

    fn parameters_ptrs(self) -> List[UnsafePointer[Tensor[dtype]]]:
        var ptrs = List[UnsafePointer[Tensor[dtype]]]()
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            if m.layer.isa[Linear[dtype]]():
                var ref l = m.layer[Linear[dtype]]
                ptrs.append(addr(l.weights))
                ptrs.append(addr(l.bias))
        return ptrs

    fn print_summary(self, input_shape: List[Int]):
        print("\nSequential Model Summary")
        print("------------------------")

        var x = Tensor[dtype].zeros(input_shape, requires_grad=False)

        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            var name = "Unknown"
            var nparams = m.num_parameters()
            var in_shape = x.shape
            var details = ""
            var trainable = "No"

            if m.layer.isa[Linear[dtype]]():
                name = "Linear"
                var ref l = m.layer[Linear[dtype]]
                details = (
                    "weight="
                    + l.weights.shape.__str__()
                    + ", bias="
                    + l.bias.shape.__str__()
                )
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
                " | I/P: ",
                in_shape,
                " -> O/P: ",
                out_shape,
                " | Params: ",
                nparams,
                (" | " + details if details != "" else ""),
                " | Trainable: ",
                trainable,
            )

        print("------------------------")
        print("Total learnable parameters: ", self.num_parameters())


fn main():
    print("passes")
