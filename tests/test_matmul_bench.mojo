from time import perf_counter_ns
from tensors import Tensor
from views import TensorView
from testing import assert_true

fn main() raises:
    size = 256

    A = Tensor.rand(size, size, requires_grad=True)
    B = Tensor.rand(size, size, requires_grad=True)

    C1, A_grad_1, B_grad_1 = bench_tensor_tensor(A, B)
    A.zero_grad()
    B.zero_grad()

    C2, A_grad_2, B_grad_2 = bench_view_view(A, B)

    assert_true(C1.all_close(C2))
    assert_true(A_grad_1.all_close(A_grad_2))
    assert_true(B_grad_1.all_close(B_grad_2))

    A.zero_grad()
    B.zero_grad()


    C3, A_grad_3, B_grad_3 = bench_tensor_view(A, B)

    assert_true(C2.all_close(C3))
    assert_true(A_grad_2.all_close(A_grad_3))
    assert_true(B_grad_2.all_close(B_grad_3))

    A.zero_grad()
    B.zero_grad()

    C4, A_grad_4, B_grad_4 = bench_view_tensor(A, B)

    assert_true(C4.all_close(C3))
    assert_true(A_grad_4.all_close(A_grad_3))
    assert_true(B_grad_4.all_close(B_grad_3))



fn bench_tensor_tensor[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], Tensor[dtype], Tensor[dtype]):
    start = perf_counter_ns()
    C = A.matmul(B)
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) -> Total: ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) backward -> Total: ", end - start)

    return C, A.grad[], B.grad[]


fn bench_view_view[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], Tensor[dtype], Tensor[dtype]):
    AV = A[:, :]
    BV = B[:, :]
    start = perf_counter_ns()
    C = AV.matmul(BV)
    end = perf_counter_ns()
    print("TensorView.matmul(TensorView) -> Total: ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("TensorView.matmul(TensorView) backward -> Total: ", end - start)

    return C, A.grad[], B.grad[]


fn bench_tensor_view[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], Tensor[dtype], Tensor[dtype]):
    BV = B[:, :]
    start = perf_counter_ns()
    C = A.matmul(BV)
    end = perf_counter_ns()
    print("Tensor.matmul(TensorView) -> Total: ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("Tensor.matmul(TensorView) backward -> Total: ", end - start)
    return C, A.grad[], B.grad[]

fn bench_view_tensor[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], Tensor[dtype], Tensor[dtype]):
    AV = A[:, :]
    start = perf_counter_ns()
    C = AV.matmul(B)
    end = perf_counter_ns()
    print("TensorView.matmul(Tensor) -> Total: ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("TensorView.matmul(Tensor) backward -> Total: ", end - start)
    return C, A.grad[], B.grad[]



