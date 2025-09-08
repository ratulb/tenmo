# layers_and_gradcheck.mojo
from tensors import Tensor
from common_utils import panic, log_debug, RED, CYAN, MAGENTA, addr
from utils import Variant
from crossentropy import CrossEntropyLoss

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
            log_debug("Freeing Linear weights - shape: " + lin.weights.shape.__str__(), RED)
            lin.weights.free()
            log_debug("Freeing Linear bias - shape: " + lin.bias.shape.__str__(), RED)
            lin.bias.free()
        elif self.layer.isa[ReLU[dtype]]():
            log_debug("ReLU has no learnable params", MAGENTA)

    fn free_all(self):
        self.free_params()

    # A convenience (but potentially unsafe) helper; prefer Sequential.parameters_ptrs()
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

    fn __init__(out self, in_features: Int, out_features: Int):
        self.weights = Tensor[dtype].rand([in_features, out_features], requires_grad=True)
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
        # store module inside container; ownership remains with container
        self.modules.append(m)

    fn __call__(self, x: Tensor[dtype]) -> Tensor[dtype]:
        var out = x
        for i in range(len(self.modules)):
            var ref m = self.modules[i]   # borrow stored module to avoid copying
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

    # ---------
    # Safe pointer collector: recomputes pointers from the modules stored in `self.modules`
    # Always use this when building optimizers or doing numerical checks.
    fn parameters_ptrs(self) -> List[UnsafePointer[Tensor[dtype]]]:
        var ptrs = List[UnsafePointer[Tensor[dtype]]]()
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            if m.layer.isa[Linear[dtype]]():
                var ref l = m.layer[Linear[dtype]]
                ptrs.append(addr(l.weights))
                ptrs.append(addr(l.bias))
            elif m.layer.isa[ReLU[dtype]]():
                # no params
                pass
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
                details = ("weight=" + l.weights.shape.__str__() + ", bias=" + l.bias.shape.__str__())
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


# --------------------
# Gradient checker (safe using parameters_ptrs)
# --------------------
fn gradcheck_param[dtype: DType = DType.float32](
    model: Sequential[dtype],
    x: Tensor[dtype],
    y: Tensor[dtype],
    criterion: CrossEntropyLoss[dtype],
    eps: Scalar[dtype] = Scalar[dtype](1e-3),
    tol: Scalar[dtype] = Scalar[dtype](1e-2),
) raises -> Bool:

    # Run forward/backward once to populate analytical gradients
    var logits = model(x)
    var loss = criterion(logits, y)
    loss.backward()

    var ok = True

    # Get pointers that are guaranteed to point into tensors owned by `model.modules`
    var ptrs = model.parameters_ptrs()

    # Sanity: print pointer info (optional debug)
    # for i in range(len(ptrs)):
    #     print("param ptr[", i, "] ->", ptrs[i])  # or a addr-to-int helper if you have one

    for idx in range(len(ptrs)):
        var p_ptr = ptrs[idx]
        # dereference pointer - this yields the actual Tensor stored in the module
        var ref p = p_ptr[]    # REF — we will mutate the underlying buffer entries
        log_debug("Gradcheck: checking param len=" + len(p.buffer).__str__(), CYAN)

        # ensure gradient storage exists
        if not p.has_grad():
            print(RED, "Gradcheck: parameter has no grad storage; skipping")
            continue

        # loop over flat buffer elements (buffer is flat)
        var n = len(p.buffer)
        for i in range(n):
            orig = p.buffer[i]

            # f(x + eps)
            p.buffer[i] = orig + eps
            var lp = criterion(model(x), y).item()

            # f(x - eps)
            p.buffer[i] = orig - eps
            var lm = criterion(model(x), y).item()

            # restore
            p.buffer[i] = orig

            grad_num = (lp - lm) / (2.0 * eps)
            grad_an = p.gradbox[].buffer[i]

            rel_err = abs(grad_an - grad_num) / (abs(grad_an) + abs(grad_num) + 1e-8)

            if rel_err > tol:
                print(
                    RED,
                    "Gradcheck FAIL param_idx=",
                    idx,
                    " elem_idx=",
                    i,
                    " an=",
                    grad_an,
                    " num=",
                    grad_num,
                    " rel_err=",
                    rel_err,
                )
                ok = False

    return ok


# --------------------
# Example test in main
# --------------------
fn test_gradcheck() raises:
    # Build model (store layers in temporaries or vars doesn't matter;
    # Sequential owns modules and parameters once appended)
    var model = Sequential()
    model.append(Linear(4, 5).into())
    model.append(ReLU().into())
    model.append(Linear(5, 3).into())

    var x = Tensor.rand([2, 4], requires_grad=True)
    var y = Tensor.d1([1, 2])

    var criterion = CrossEntropyLoss()

    var passed = gradcheck_param(model, x, y, criterion, Scalar[DType.float32](1e-3), Scalar[DType.float32](1e-2))
    if passed:
        print(CYAN, "Gradient check PASSED ✅")
    else:
        print(RED, "Gradient check FAILED ❌")

    # clean up
    _ = model


fn main() raises:
    test_gradcheck()

