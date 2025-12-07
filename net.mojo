from tenmo import Tensor
from shapes import Shape
from gradbox import Gradbox
from math import sqrt
from common_utils import addr, panic
from utils import Variant
from forwards import Matmul, Adder, Multiplicator, Subtractor, Clip
from operators import mm, mv, vm, dot


@fieldwise_init
struct Linear[dtype: DType, mode: Int = mm](
    ImplicitlyCopyable & Movable
):  # alias mode = mm  # tensor & tensor matmul
    """Fully connected layer: y = xW + b."""

    var weight: Tensor[Self.dtype]
    var bias: Tensor[Self.dtype]
    var in_features: Int
    var out_features: Int
    var training: Bool  # Training mode flag

    fn __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        xavier: Bool = True,
        bias_noise: Scalar[Self.dtype] = Scalar[Self.dtype](0),
        weight_factor: Scalar[Self.dtype] = Scalar[Self.dtype](1),
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.training = True  # Default to training mode

        if xavier:
            limit = Scalar[Self.dtype](sqrt(6.0 / (in_features + out_features)))
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=Shape(in_features, out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            self.bias = (
                Tensor[Self.dtype].rand(Shape(out_features), requires_grad=True)
                * bias_noise
            )
        else:
            var std = Scalar[Self.dtype](sqrt(2.0 / in_features))
            self.weight = (
                Tensor[Self.dtype].randn(
                    shape=Shape(in_features, out_features),
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * std
            )
            self.bias = (
                Tensor[Self.dtype].randn(
                    Shape(out_features), init_seed=init_seed, requires_grad=True
                )
                * bias_noise
            )

    fn __call__(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        ref xs_shape = xs.shape()
        ref weight_shape = self.weight.shape()

        if xs_shape[-1] != weight_shape[0]:
            panic(
                "Linear forward: input dim mismatch: input shape → ",
                xs_shape.__str__(),
                "and  weights shape → ",
                weight_shape.__str__(),
            )

        # Branch on training mode - compiler will optimize each path separately
        if self.training:
            # Training mode: build computational graph
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=mode
            ](  # mode == mm matrix mul
                xs, self.weight
            )
            return Adder[Self.dtype].forward[track_grad=True](
                matmul_out, self.bias
            )
        else:
            # Eval mode: no graph building - pure computation
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=mode
            ](  # mode == mm matrix mul
                xs, self.weight
            )
            return Adder[Self.dtype].forward[track_grad=False](
                matmul_out, self.bias
            )

    fn parameters(ref self) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_any_origin()
        )
        params.append(
            UnsafePointer(to=self.bias)
            .unsafe_mut_cast[True]()
            .as_any_origin()
        )

        return params^

    fn num_parameters(self) -> Int:
        return self.weight.numels() + self.bias.numels()

    fn train(mut self):
        """Set to training mode - enables gradient tracking."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


@register_passable
struct ReLU[dtype: DType](ImplicitlyCopyable):
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return x.relu[track_grad=True]()
        else:
            return x.relu[track_grad=False]()

    fn parameters(ref self) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


@register_passable
struct Sigmoid[dtype: DType](ImplicitlyCopyable):
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return x.sigmoid[track_grad=True]()
        else:
            return x.sigmoid[track_grad=False]()

    fn parameters(ref self) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


@register_passable
struct Tanh[dtype: DType](ImplicitlyCopyable):
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return x.tanh[track_grad=True]()
        else:
            return x.tanh[track_grad=False]()

    fn parameters(ref self) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


# Refer to operators & matmul
# Defined in operators

# alias dot = 28  # dot product
# alias vm = 29  # vector & tensor matmul
# alias mv = 30  # tensor & vector matmul
# alias mm = 31  # tensor & tensor matmul

alias Layer[dtype: DType] = Variant[
    Linear[dtype, mm],
    ReLU[dtype],
    Sigmoid[dtype],
    Tanh[dtype],
]


@fieldwise_init
struct Module[dtype: DType](ImplicitlyCopyable & Movable):
    var layer: Layer[Self.dtype]

    fn __call__(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.layer.isa[Linear[Self.dtype, mm]]():
            return self.layer[Linear[Self.dtype, mm]](xs)
        elif self.layer.isa[ReLU[Self.dtype]]():
            return self.layer[ReLU[Self.dtype]](xs)
        elif self.layer.isa[Sigmoid[Self.dtype]]():
            return self.layer[Sigmoid[Self.dtype]](xs)
        elif self.layer.isa[Tanh[Self.dtype]]():
            return self.layer[Tanh[Self.dtype]](xs)

        else:
            panic("Unknown module type")
            return Tensor[Self.dtype].scalar(0)

    fn parameters(ref self) -> List[
        UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]
    ]:
        if self.layer.isa[Linear[Self.dtype]]():
            return self.layer[Linear[Self.dtype]].parameters()
        else:
            return List[
                UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]
            ]()

    fn num_parameters(self) -> Int:
        if self.layer.isa[Linear[Self.dtype, mm]]():
            return self.layer[Linear[Self.dtype, mm]].num_parameters()
        elif self.layer.isa[ReLU[Self.dtype]]():
            return self.layer[ReLU[Self.dtype]].num_parameters()
        elif self.layer.isa[Sigmoid[Self.dtype]]():
            return self.layer[Sigmoid[Self.dtype]].num_parameters()
        elif self.layer.isa[Tanh[Self.dtype]]():
            return self.layer[Tanh[Self.dtype]].num_parameters()

        else:
            return 0

    fn zero_grad(self):
        """Zero all parameter gradients."""
        for parameter in self.parameters():
            parameter[].zero_grad()

    fn train(mut self):
        """Set module to training mode."""
        if self.layer.isa[Linear[Self.dtype, mm]]():
            self.layer[Linear[Self.dtype, mm]].train()
        elif self.layer.isa[ReLU[Self.dtype]]():
            self.layer[ReLU[Self.dtype]].train()
        elif self.layer.isa[Sigmoid[Self.dtype]]():
            self.layer[Sigmoid[Self.dtype]].train()
        elif self.layer.isa[Tanh[Self.dtype]]():
            self.layer[Tanh[Self.dtype]].train()

    fn eval(mut self):
        """Set module to evaluation mode."""
        if self.layer.isa[Linear[Self.dtype]]():
            self.layer[Linear[Self.dtype]].eval()
        elif self.layer.isa[ReLU[Self.dtype]]():
            self.layer[ReLU[Self.dtype]].eval()
        elif self.layer.isa[Sigmoid[Self.dtype]]():
            self.layer[Sigmoid[Self.dtype]].eval()
        elif self.layer.isa[Tanh[Self.dtype]]():
            self.layer[Tanh[Self.dtype]].eval()


@fieldwise_init
struct Sequential[dtype: DType](Copyable & Movable):
    var modules: List[Module[Self.dtype]]

    fn __init__(out self):
        self.modules = List[Module[Self.dtype]]()

    fn append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            self.modules.append(m)

    fn __call__(mut self, xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var out = xs
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
            out = m(out)
        return out

    fn parameters(ref self) -> List[
        UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]
    ]:
        var params = List[
            UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]
        ]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    fn num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    fn train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    fn eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()


# -----------------------------------------
# Mean Squared Error Loss
# -----------------------------------------
@fieldwise_init
@register_passable
struct MSELoss[dtype: DType = DType.float32]:
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __call__(
        self, preds: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype]:
        if self.training:
            return preds.mse[track_grad=True](target)
        else:
            return preds.mse[track_grad=False](target)

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
@register_passable
struct BCELoss[dtype: DType = DType.float32]:
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    fn __init__(out self, epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9)):
        self.training = True
        self.epsilon = epsilon

    # Instance method - respects training mode
    fn __call__(
        self, pred: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](pred, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](pred, target, self.epsilon)

    # Static method - can be called directly or via instance
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype]:
        # Clip for numerical stability
        var pred_safe = Clip[Self.dtype].forward[track_grad](
            pred, epsilon, 1 - epsilon
        )

        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        var log_pred = pred_safe.log[track_grad]()
        var term1 = Multiplicator[Self.dtype].forward[track_grad](
            target, log_pred
        )

        var one = Tensor[Self.dtype].scalar(1)
        var one_minus_target = Subtractor[Self.dtype].forward[track_grad](
            one, target
        )
        var one_minus_pred = Subtractor[Self.dtype].forward[track_grad](
            one, pred_safe
        )
        var log_one_minus_pred = one_minus_pred.log[track_grad]()
        var term2 = Multiplicator[Self.dtype].forward[track_grad](
            one_minus_target, log_one_minus_pred
        )

        var sum_terms = Adder[Self.dtype].forward[track_grad](term1, term2)
        var neg_one = Tensor[Self.dtype].scalar(-1)
        var loss = Multiplicator[Self.dtype].forward[track_grad](
            sum_terms, neg_one
        )

        return loss.mean[track_grad]()

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
@register_passable
struct BCEWithLogitsLoss[dtype: DType = DType.float32]:
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    fn __init__(out self, epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9)):
        self.training = True
        self.epsilon = epsilon

    # Instance method - respects training mode
    fn __call__(
        self, logits: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](logits, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](logits, target, self.epsilon)

    # Static method - can be called directly or via instance
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype]:
        # Apply sigmoid to convert logits to probabilities
        var pred_probs = logits.sigmoid[track_grad]()

        # Clip for numerical stability
        var probs_safe = Clip[Self.dtype].forward[track_grad](
            pred_probs, epsilon, 1 - epsilon
        )

        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        var log_probs = probs_safe.log[track_grad]()
        var term1 = Multiplicator[Self.dtype].forward[track_grad](
            target, log_probs
        )

        var one = Tensor[Self.dtype].scalar(1)
        var one_minus_target = Subtractor[Self.dtype].forward[track_grad](
            one, target
        )
        var one_minus_probs = Subtractor[Self.dtype].forward[track_grad](
            one, probs_safe
        )
        var log_one_minus = one_minus_probs.log[track_grad]()
        var term2 = Multiplicator[Self.dtype].forward[track_grad](
            one_minus_target, log_one_minus
        )

        var sum_terms = Adder[Self.dtype].forward[track_grad](term1, term2)
        var neg_one = Tensor[Self.dtype].scalar(-1)
        var loss = Multiplicator[Self.dtype].forward[track_grad](
            sum_terms, neg_one
        )

        return loss.mean[track_grad]()

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
struct SGD[dtype: DType, //](ImplicitlyCopyable & Movable):
    """Stochastic Gradient Descent."""

    var parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]
    var lr: Scalar[Self.dtype]
    var momentum: Scalar[Self.dtype]
    var velocities: List[Gradbox[Self.dtype]]  # For momentum

    fn __init__(
        out self,
        parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]],
        lr: Scalar[Self.dtype] = 0.01,
        momentum: Scalar[Self.dtype] = 0.0,
    ):
        self.parameters = parameters.copy()
        self.lr = lr
        self.momentum = momentum
        self.velocities = List[Gradbox[Self.dtype]]()

        # Initialize velocities
        if momentum > 0:
            for parameter in self.parameters:
                self.velocities.append(
                    Gradbox[Self.dtype].zeros(parameter[].shape(), share=False)
                )

    fn __copyinit__(out self, existing: Self):
        self.parameters = existing.parameters.copy()
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.velocities = existing.velocities.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.parameters = existing.parameters^
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.velocities = existing.velocities^

    fn step(mut self):
        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if parameter.requires_grad and parameter.has_grad():
                ref grad = parameter.gradients()[]
                if self.momentum > 0:
                    # v = momentum * v + grad
                    self.velocities[i] = (
                        self.velocities[i] * self.momentum + grad
                    )
                    # param -= lr * v
                    parameter -= self.velocities[i] * self.lr
                else:
                    # param -= lr * grad
                    parameter -= grad * self.lr

    fn zero_grad(self):
        for parameter in self.parameters:
            parameter[].zero_grad()

    fn set_lr(mut self, lr: Scalar[Self.dtype]):
        self.lr = lr


fn main():
    pass
