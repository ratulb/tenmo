# gradcheck.mojo
from tensors import Tensor
from common_utils import log_debug, RED, CYAN
from crossentropy import CrossEntropyLoss
from layers import Sequential, Linear, ReLU


# --------------------
# Gradient checker (pluggable)
# --------------------
fn gradcheck_param[
    dtype: DType = DType.float32
](
    model: Sequential[dtype],
    x: Tensor[dtype],
    y: Tensor[dtype],
    criterion: CrossEntropyLoss[dtype],
    eps: Scalar[dtype] = Scalar[dtype](1e-3),
    tol: Scalar[dtype] = Scalar[dtype](1e-2),
) raises -> Bool:
    var logits = model(x)
    var loss = criterion(logits, y)
    loss.backward()

    var ok = True
    var ptrs = model.parameters_ptrs()

    for idx in range(len(ptrs)):
        var p_ptr = ptrs[idx]
        var ref p = p_ptr[]
        log_debug(
            "Gradcheck: checking param len=" + len(p.buffer).__str__(), CYAN
        )

        if not p.has_grad():
            print(RED, "Gradcheck: parameter has no grad storage; skipping")
            continue

        var n = len(p.buffer)
        for i in range(n):
            orig = p.buffer[i]

            p.buffer[i] = orig + eps
            var lp = criterion(model(x), y).item()

            p.buffer[i] = orig - eps
            var lm = criterion(model(x), y).item()

            p.buffer[i] = orig

            grad_num = (lp - lm) / (2.0 * eps)
            grad_an = p.gradbox[].buffer[i]

            rel_err = abs(grad_an - grad_num) / (
                abs(grad_an) + abs(grad_num) + 1e-8
            )

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
# Example usage
# --------------------
fn test_gradcheck() raises:
    var model = Sequential()
    model.append(Linear(4, 5).into())
    model.append(ReLU().into())
    model.append(Linear(5, 3).into())

    var x = Tensor.rand([2, 4], requires_grad=True)
    var y = Tensor.d1([1, 2])

    var criterion = CrossEntropyLoss()

    var passed = gradcheck_param(model, x, y, criterion)
    if passed:
        print(CYAN, "Gradient check PASSED ✅")
    else:
        print(RED, "Gradient check FAILED ❌")

    _ = model


fn main() raises:
    test_gradcheck()
