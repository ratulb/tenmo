from tenmo.tensor import Tensor
from std.testing import assert_equal


# =====================================================================
# Toy: parameterize over target_dtype, default DType.int64
# =====================================================================

struct Loss[
    logit_dtype: DType,
    target_dtype: DType = DType.int64,
](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    def forward(
        logits: Tensor[Self.logit_dtype],
        target: Tensor[Self.target_dtype],
    ) -> Tensor[Self.logit_dtype]:
        # Toy: sum over relevant positions — no element-wise cast needed
        var total = Scalar[Self.logit_dtype](0)
        var n = target.numels()
        for i in range(n):
            var idx = target[i].__int__()
            total += logits[idx]
        return Tensor[Self.logit_dtype].scalar(total)


# =====================================================================
# Test: default int64
# =====================================================================

def test_default_int64() raises:
    var logits = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0])
    var target = Tensor[DType.int64].d1([0, 2, 3])
    var result = Loss[DType.float32].forward(logits, target)
    assert_equal(result.item(), 80.0)
    print("  default int64 OK")


# =====================================================================
# Test: explicit int32
# =====================================================================

def test_explicit_int32() raises:
    var logits = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0])
    var target = Tensor[DType.int32].d1([1, 1, 0])
    var result = Loss[DType.float32, DType.int32].forward(logits, target)
    assert_equal(result.item(), 50.0)
    print("  explicit int32 OK")


# =====================================================================
# Test: mixing int32 target with default int64 — COMPILE ERROR
# =====================================================================

# Uncommenting this will FAIL at compile time:
#
#   var logits = Tensor[DType.float32].d1([10.0, 20.0, 30.0])
#   var target = Tensor[DType.int32].d1([0, 1, 2])
#   var result = Loss[DType.float32].forward(logits, target)
#
# Error: can't implicitly convert Tensor[DType.int32] to Tensor[DType.int64]


# =====================================================================
# Test: same Loss struct, different target_dtype — compiled once per dtype
# =====================================================================

def test_both_instantiations() raises:
    var logits = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0])

    var t_i64 = Tensor[DType.int64].d1([0, 1, 2])
    var out_i64 = Loss[DType.float32, DType.int64].forward(logits, t_i64)
    assert_equal(out_i64.item(), 60.0)

    var t_i32 = Tensor[DType.int32].d1([0, 1, 2])
    var out_i32 = Loss[DType.float32, DType.int32].forward(logits, t_i32)
    assert_equal(out_i32.item(), 60.0)

    print("  both instantiations OK")


# =====================================================================
# Test: to_dtype cast shows the overhead we're avoiding
# =====================================================================

def test_to_dtype_overhead() raises:
    var logits = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0])
    var target = Tensor[DType.int32].d1([0, 2, 3])
    # This is what to_dtype does — iterates all elements:
    var cast_target = target.to_dtype[DType.int64]()
    # This copies every element — wasteful
    var result = Loss[DType.float32].forward(logits, cast_target)
    assert_equal(result.item(), 80.0)
    print("  to_dtype works but copies elements — wasteful")


def main() raises:
    print("")
    test_default_int64()
    test_explicit_int32()
    test_both_instantiations()
    test_to_dtype_overhead()
    print("\nAll OK")
