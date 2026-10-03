from .tensor import Tensor
from .shared.shapes import Shape
from std.math import sqrt
from .shared.panic import panic
from .shared import WeightStrategy
from .gpu.device import Device

comptime Strategy_Normal = WeightStrategy(0)
comptime Strategy_Uniform = WeightStrategy(1)
comptime Strategy_Xavier = WeightStrategy(2)
comptime Strategy_Kaiming = WeightStrategy(3)
comptime Strategy_Zero = WeightStrategy(4)


struct Weights[dtype: DType]:
    @staticmethod
    def initialize(
        weight_shape: Shape,
        fan_in: Int,
        fan_out: Int,
        init_seed: Optional[Int] = None,
        init_method: WeightStrategy = "normal",
        requires_grad: Bool = True,
        bias_len: Int = 0,
        bias_zero: Bool = True,
        device: Optional[Device] = None,
    ) -> Tuple[Tensor[Self.dtype], Optional[Tensor[Self.dtype]]]:
        """Weight initializer.
        Args:
            weight_shape:  Shape of the weight tensor (any rank).
            fan_in:        Caller-computed fan-in (Linear: in_features;
                           Conv2D: in_channels*kH*kW).
            fan_out:       Caller-computed fan-out (Linear: out_features;
                           Conv2D: out_channels*kH*kW).
            init_seed:     Optional init seed
            init_method:   Weight init strategy:
                            "normal"   — N(0, 1) (PyTorch default)
                            "uniform"  — U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                            "xavier"   — Xavier uniform
                            "kaiming/he"  — Kaiming normal
                            "zero"  — weights initialized to 0.
                            See WeightStrategy.
            bias_len:      Length of the 1D bias (0 = no bias).
            bias_zero:     Should bias be set to zero.

        Choosing a scheme — standalone layer defaults differ by role
        (`Linear`/`LinearBLAS` → "uniform", `Embedding` → "normal",
        `PositionalEmbedding`/`MLP` → "xavier", `Conv2D` → "he"), but
        containers (attention blocks, GPTEmbedding, GPTModel) FORWARD one
        `init_method` down their whole subtree, so a default `GPTModel` is
        all-xavier end to end. Keep it that way: the one hard rule is
        that every layer in one model must draw from this ONE vocabulary. A name outside it
        (`WeightStrategy` panics on unknown strings) fails LOUDLY at
        construction; a name silently unhandled somewhere builds a
        dead-zero model whose `backward()` then faithfully propagates
        zeros — which looks exactly like a severed autograd graph and
        costs a full bisection to exonerate (see tests/test_gpt_gradflow).
        """
        var weight: Tensor[Self.dtype]
        var bias_tensor: Optional[Tensor[Self.dtype]] = None

        if Strategy_Normal == init_method:
            # "normal" — PyTorch convention: N(0, 1)
            # (Some implementations instead use 1/√in_features as std.
            # but Gaussian instead of uniform — either is defensible.
            weight = Tensor[Self.dtype].randn(
                weight_shape,
                mean=0.0,
                std=1.0,
                init_seed=init_seed,
                requires_grad=requires_grad,
                device=device,
            )
            if bias_len > 0:
                if not bias_zero:
                    bias_tensor = Tensor[Self.dtype].randn(
                        Shape(bias_len),
                        mean=0.0,
                        std=1.0,
                        init_seed=init_seed,
                        requires_grad=requires_grad,
                        device=device,
                    )

                else:
                    bias_tensor = Tensor[Self.dtype].zeros(
                        Shape(bias_len),
                        requires_grad=requires_grad,
                        device=device,
                    )

        elif Strategy_Uniform == init_method:
            # "uniform" — PyTorch nn.Linear's default: U(-1/√fan_in, 1/√fan_in)
            var limit = Scalar[Self.dtype](1.0 / sqrt(Float64(fan_in)))
            weight = Tensor[Self.dtype].rand(
                weight_shape,
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=requires_grad,
                device=device,
            )
            if bias_len > 0:
                if not bias_zero:
                    bias_tensor = Tensor[Self.dtype].rand(
                        Shape(bias_len),
                        min=-limit,
                        max=limit,
                        init_seed=init_seed,
                        requires_grad=requires_grad,
                        device=device,
                    )

                else:
                    bias_tensor = Tensor[Self.dtype].zeros(
                        Shape(bias_len),
                        requires_grad=requires_grad,
                        device=device,
                    )
        elif Strategy_Xavier == init_method:
            # "xavier" (Glorot uniform) — balances variance across fan_in AND fan_out,
            # intended for tanh/sigmoid-like symmetric activations:
            #   U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(fan_in + fan_out))
            )

            weight = Tensor[Self.dtype].rand(
                weight_shape,
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=requires_grad,
                device=device,
            )
            if bias_len > 0:
                if not bias_zero:
                    bias_tensor = Tensor[Self.dtype].rand(
                        Shape(bias_len),
                        min=-limit,
                        max=limit,
                        init_seed=init_seed,
                        requires_grad=requires_grad,
                        device=device,
                    )

                else:
                    bias_tensor = Tensor[Self.dtype].zeros(
                        Shape(bias_len),
                        requires_grad=requires_grad,
                        device=device,
                    )
        elif Strategy_Kaiming == init_method:
            # "he" (Kaiming) — accounts only for fan_in, intended for ReLU/GELU-like
            # activations that zero out roughly half their input, so need extra
            # variance to compensate:
            #   normal:  N(0, √(2/fan_in))
            #   uniform: U(-√(6/fan_in), √(6/fan_in))
            # Normal is the more common "He init" convention though either can be picked

            var std = sqrt(2.0 / Float64(fan_in))
            weight = Tensor[Self.dtype].randn(
                weight_shape,
                mean=0.0,
                std=std,
                init_seed=init_seed,
                requires_grad=requires_grad,
                device=device,
            )
            if bias_len > 0:
                if not bias_zero:
                    bias_tensor = Tensor[Self.dtype].randn(
                        Shape(bias_len),
                        mean=0.0,
                        std=std,
                        init_seed=init_seed,
                        requires_grad=requires_grad,
                        device=device,
                    )

                else:
                    bias_tensor = Tensor[Self.dtype].zeros(
                        Shape(bias_len),
                        requires_grad=requires_grad,
                        device=device,
                    )

        else:  # Zero
            weight = Tensor[Self.dtype].zeros(
                weight_shape,
                requires_grad=requires_grad,
                device=device,
            )
            if bias_len > 0:
                if not bias_zero:
                    bias_tensor = Tensor[Self.dtype].randn(
                        Shape(bias_len),
                        mean=0.0,
                        std=1.0,
                        init_seed=init_seed,
                        requires_grad=requires_grad,
                        device=device,
                    )

                else:
                    bias_tensor = Tensor[Self.dtype].zeros(
                        Shape(bias_len),
                        requires_grad=requires_grad,
                        device=device,
                    )

        return weight, bias_tensor

    @staticmethod
    def initialize(
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: WeightStrategy = "normal",
        requires_grad: Bool = True,
        bias: Bool = True,
        bias_zero: Bool = True,
        device: Optional[Device] = None,
    ) -> Tuple[Tensor[Self.dtype], Optional[Tensor[Self.dtype]]]:
        """2D convenience wrapper.
        Builds an `(in_features, out_features)` weight with
        fan_in=in_features, fan_out=out_features; delegates to the
        Shape-based overload above."""
        return Self.initialize(
            Shape(in_features, out_features),
            in_features,
            out_features,
            init_seed=init_seed,
            init_method=init_method,
            requires_grad=requires_grad,
            bias_len=out_features if bias else 0,
            bias_zero=bias_zero,
            device=device,
        )
