from memory import UnsafePointer
from tensors import Tensor
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape
from os import abort

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores

alias Noop = 0
alias AddScalar = 1
alias SubtractScalar = 2
alias MulScalar = 3
alias MulTensor = 4
alias AddTensor = 5
alias SubtractTensor = 6


fn sum_3d[
    axis: Int,
    dtype: DType = DType.float32,
    simd_width: Int = simdwidthof[dtype](),
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    constrained[
        axis == 0 or axis == 1 or axis == 2,
        "operators -> sum_3d - axis can be only 0, 1 or 2. axis: "
        + String(axis),
    ]()
    if tensor.ndim() != 3:
        abort(
            "operators -> sum_3d - Tensor is ndim is not 3: "
            + String(tensor.ndim())
        )
    tensor.print()
    print()
    shape = tensor.shape
    H = tensor.shape[0]
    W = tensor.shape[1]
    D = tensor.shape[2]

    fn offset_for_xy(x: Int, y: Int) -> Int:
        return x * (W * D) + y * D

    fn offset_for_xz(x: Int, z: Int) -> Int:
        return x * (W * D) + z

    fn offset_for_yz(y: Int, z: Int) -> Int:
        return (y * D) + z

    out = Tensor[dtype].zeros(
        shape.drop_axis(axis), requires_grad=tensor.requires_grad
    )
    if axis == 2:
        for indices in out.shape:
            x, y = indices[0], indices[1]
            offset = offset_for_xy(x, y)
            for idx in range(D):
                out[indices] += tensor.data.load[width=1](offset + idx)
    if axis == 1:
        for indices in out.shape:
            x, z = indices[0], indices[1]
            offset = offset_for_xz(x, z)
            for idx in range(W):
                out[indices] += tensor.data.load[width=1](offset + idx * D)
    if axis == 0:
        for indices in out.shape:
            y, z = indices[0], indices[1]
            offset = offset_for_yz(y, z)
            for idx in range(H):
                out[indices] += tensor.data.load[width=1](offset + idx * W * D)

    return out


fn sum_across_rows[  # sum axis=1
    dtype: DType = DType.float32, simd_width: Int = simdwidthof[dtype]()
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    ROWS = tensor.shape[0]
    COLS = tensor.shape[1]
    shape = Shape.of(ROWS)
    out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    # num_threads = num_physical_cores()
    num_threads = num_logical_cores()
    chunk_size = (ROWS + num_threads - 1) // num_threads

    @parameter
    fn sum_chunk(thread_id: Int):
        start = thread_id * chunk_size
        end = min(start + chunk_size, ROWS)

        for row in range(start, end):
            row_start = tensor.data.offset(row * COLS)

            @parameter
            fn sum_row_elems[_simd_width: Int](idx: Int):
                out[row] += row_start.load[width=_simd_width](idx).reduce_add()

            vectorize[sum_row_elems, simd_width](COLS)

    parallelize[sum_chunk](num_threads)
    return out


fn sum_1d[
    dtype: DType = DType.float32, simd_width: Int = simdwidthof[dtype]()
](tensor: Tensor[dtype], start_index: Int = 0) -> Tensor[dtype]:
    if tensor.ndim() != 1:
        abort("operators -> sum_1d - tensor ndim is not 1")
    start = start_index if start_index >= 0 else len(tensor) + start_index
    if not (start >= 0 and start < len(tensor)):
        abort(
            "operators -> sum_1d start_index out of bounds: "
            + String(start_index)
        )
    out = Tensor[dtype].zeros(1, requires_grad=tensor.requires_grad)
    accum: Scalar[dtype] = 0

    @parameter
    fn sum_from[_simd_width: Int](idx: Int):
        accum += tensor.data.load[width=_simd_width](idx + start).reduce_add()

    vectorize[sum_from, simd_width](tensor.numels() - start)
    out[0] = accum
    return out


fn sum_across_cols[  # sum axis = 0
    # dtype: DType = DType.float32
    dtype: DType,
    //
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    COLS = tensor.shape[1]
    ROWS = tensor.shape[0]
    alias simd_width = simdwidthof[dtype]()
    shape = Shape.of(COLS)
    var out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    num_threads = num_physical_cores()
    chunk_size = (COLS + num_threads - 1) // num_threads

    @parameter
    fn sum_chunk(thread_id: Int):
        start = thread_id * chunk_size
        end = min(start + chunk_size, COLS)

        for col in range(start, end):
            base_ptr = tensor.data.offset(col)
            var summ: Scalar[dtype] = 0

            var row = 0
            while row + simd_width <= ROWS:
                simd_val = base_ptr.offset(row * COLS).load[width=simd_width]()
                summ += simd_val.reduce_add()
                row += simd_width

            # Handle tail (remaining rows not divisible by simd_width)
            while row < ROWS:
                summ += base_ptr.offset(row * COLS).load()
                row += 1

            out[col] = summ

    parallelize[sum_chunk](num_threads)
    return out


_ = """fn sum_across_cols[# sum axis = 0
    dtype: DType = DType.float32
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    COLS = tensor.shape[1]
    shape = Shape.of(COLS)
    out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    @parameter
    fn sum_cols(col: Int):
        col_start = tensor.data.offset(col)
        out[col] += col_start.strided_load[width = simdwidthof[dtype]()](
            COLS
        ).reduce_add()

    parallelize[sum_cols](COLS)
    return out
fn sum_across_cols[# sum axis = 0
    dtype: DType = DType.float32
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    COLS = tensor.shape[1]
    alias simd_width = simdwidthof[dtype]()
    shape = Shape.of(COLS)
    var out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    num_threads = num_logical_cores()
    chunk_size = (COLS + num_threads - 1) // num_threads

    @parameter
    fn sum_chunk(thread_id: Int):
        start = thread_id * chunk_size
        end = min(start + chunk_size, COLS)

        for col in range(start, end):
            col_start = tensor.data.offset(col)
            out[col] += col_start.strided_load[width = simd_width](COLS).reduce_add()

    parallelize[sum_chunk](num_threads)
    return out"""


# Element wise operatorns
fn __tensor_op_tensor__[
    dtype: DType, op: Int
](this: Tensor[dtype], that: Tensor[dtype]) -> Tensor[dtype]:
    if this.shape != that.shape:
        abort(
            "operator ->__tensor_op__tensor("
            + String(op)
            + ")  -> Dimension mismatch: "
            + this.shape.__str__()
            + " <=>"
            + that.shape.__str__()
        )
    requires_grad = this.requires_grad or that.requires_grad
    out = Tensor[dtype](this.shape, requires_grad)

    @parameter
    fn mul_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                * that.data.load[width=simd_width](idx)
            ),
        )

    @parameter
    fn add_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                + that.data.load[width=simd_width](idx)
            ),
        )

    @parameter
    fn subtract_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                - that.data.load[width=simd_width](idx)
            ),
        )

    if op == MulTensor:
        vectorize[mul_elems, simdwidthof[dtype]()](out.numels())
    elif op == AddTensor:
        vectorize[add_elems, simdwidthof[dtype]()](out.numels())
    elif op == SubtractTensor:
        vectorize[subtract_elems, simdwidthof[dtype]()](out.numels())

    return out


# Tensor and scalar ops
fn __tensor_op_scalar__[
    dtype: DType, op: Int
](this: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
    var out = Tensor[dtype](this.shape, this.requires_grad)

    @parameter
    fn add_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) + scalar
        )

    @parameter
    fn subtract_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) - scalar
        )

    @parameter
    fn mul_by_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) * scalar
        )

    if op == MulScalar:
        vectorize[mul_by_scalar, simdwidthof[dtype]()](out.numels())
    elif op == AddScalar:
        vectorize[add_scalar, simdwidthof[dtype]()](out.numels())
    elif op == SubtractScalar:
        vectorize[subtract_scalar, simdwidthof[dtype]()](out.numels())

    return out


from testing import assert_true


fn test_sum_1d() raises:
    tensor = Tensor.of(1, 2, 3, 4, 5, 6, 7, 8)
    summ = sum_1d(tensor)
    assert_true(summ.item() == 36, "sum_1d assertion failed")

    summ = sum_1d(tensor, 1)
    assert_true(summ.item() == 35, "sum_1d start_index = 1 assertion failed")

    summ = sum_1d(tensor, -8)
    assert_true(summ.item() == 36.0, "sum_1d start_index = -8 assertion failed")

    summ = sum_1d(tensor, -1)
    assert_true(summ.item() == 8, "sum_1d start_index = -1 assertion failed")
    Tensor.free_all(tensor)


fn test_sum_across_rows() raises:
    tensor = Tensor.of[5](
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    )
    summ = sum_across_rows(tensor)
    expect = Tensor.of(15.0, 40.0, 65.0)
    assert_true(
        (summ == expect).all_true(),
        "operators -> sum_across_rows assertion failed",
    )
    Tensor.free_all(summ, tensor)


fn test_sum_across_cols() raises:
    tensor = Tensor.of[5](
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    )
    summ = sum_across_cols(tensor)
    expect = Tensor.of(18.0, 21.0, 24.0, 27.0, 30.0)
    assert_true(
        (summ == expect).all_true(),
        "operators -> sum_across_cols assertion failed",
    )
    Tensor.free_all(summ, tensor)


fn main() raises:
    test_sum_across_rows()
    test_sum_1d()
    test_sum_across_cols()
    test_sum_1d()
    tensor_3d = Tensor.rand(2, 3, 4)
    # _result = sum_3d[2](tensor_3d)
    # result.print()
    # _result = sum_3d[1](tensor_3d)
    # _result.print()
    result = sum_3d[0](tensor_3d)
    result.print()
