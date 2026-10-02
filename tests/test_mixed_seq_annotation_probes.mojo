"""Guard-probe harness for net.MixedSequential.forward annotations.

Each `--probe-*` invocation performs exactly ONE mismatched call; the
runtime annotation guard panics (→ abort) with its diagnostic, and the
spawning test in tests/test_mixed_sequential.mojo asserts non-zero exit
plus the exact message. Deliberately a MINIMAL file: the child JIT runs
alongside its resident parent, so its compile surface must stay small
enough to avoid OOM (the full-suite file was killed at ~10 GB RSS).
"""

from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.net import Linear, ReLU
from tenmo.net import MixedSequential
from tenmo.shared.panic import panic


def make_probe_model[seed: Int]() -> MixedSequential:
    """F32 → [cast] → f64 chain, same shape as the equivalence fixture."""
    comptime f32 = DType.float32
    comptime f64 = DType.float64
    var model = MixedSequential()
    model.append(Linear[f32](4, 6, init_seed=seed))
    model.append(ReLU[f32]())
    model.append(Linear[f64](6, 3, init_seed=seed))
    model.append(ReLU[f64]())
    return model^


def fire_head() raises:
    """F64 input offered to an f32-headed chain: head guard must abort."""
    var model = make_probe_model[51]()
    var x = Tensor[DType.float64].full(Shape(2, 4), 0.5)
    _ = model.forward[DType.float64, DType.float64](x)
    panic("head annotation guard did not fire")


def fire_tail() raises:
    """Correctly-typed input, but Out annotated f32 against an f64 tail."""
    var model = make_probe_model[52]()
    var x = Tensor[DType.float32].full(Shape(2, 4), 0.5)
    _ = model.forward[DType.float32, DType.float32](x)
    panic("tail annotation guard did not fire")


def fire_empty() raises:
    """Forward through a container with zero records."""
    var model = MixedSequential()
    var x = Tensor[DType.float32].full(Shape(2, 4), 0.5)
    _ = model.forward[DType.float32, DType.float32](x)
    panic("empty-model guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-head":
            fire_head()
            return
        elif args[i] == "--probe-tail":
            fire_tail()
            return
        elif args[i] == "--probe-empty":
            fire_empty()
            return
