from memory import UnsafePointer
from tensors import Tensor
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores

alias Noop = 0
alias AddScalar = 1
alias SubtractScalar = 2
alias MulScalar = 3
alias MulTensor = 4
alias AddTensor = 5
alias SubtractTensor = 6


fn sum_across_rows[  # sum axis=1
    dtype: DType = DType.float32, simd_width: Int = simdwidthof[dtype]()
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    ROWS = tensor.shape[0]
    COLS = tensor.shape[1]
    shape = Shape.of(ROWS)
    out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    #num_threads = num_physical_cores()
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


fn sum_across_cols[  # sum axis = 0
    dtype: DType = DType.float32
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
](this: Tensor[dtype], that: Tensor[dtype]) raises -> Tensor[dtype]:
    if this.shape != that.shape:
        raise Error(
            "__tensor_op__tensor(" + String(op) + ")  -> Dimension mismatch:",
            this.shape,
            that.shape,
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
](this: Tensor[dtype], scalar: Scalar[dtype]) raises -> Tensor[dtype]:
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


fn main() raises:
    tensor = Tensor.of[5](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    tensor.print()
    print()
    # summ = sum_across_rows(tensor)
    # summ.print()
    # sl = tensor.data.strided_load[width=16](5)
    # print(sl.__str__())
    # print(len(sl))
    # print(sl.reduce_add().__str__())
    # summ = sum_across_cols(tensor)
    # summ.print()
    print()
    # print(num_logical_cores(), num_physical_cores())
    summ = sum_across_rows(tensor)
    summ.print()
