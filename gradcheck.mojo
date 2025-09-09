from tensors import Tensor
from common_utils import log_debug, RED, CYAN
from layers import Sequential


# --------------------
# Gradient checker (safe using parameters_ptrs)
# --------------------
fn gradcheck_param[
    dtype: DType = DType.float32
](
    model: Sequential[dtype],
    x: Tensor[dtype],
    y: Tensor[dtype],
    loss_fn: fn (Tensor[dtype], Tensor[dtype]) -> Tensor[dtype],
    eps: Scalar[dtype] = Scalar[dtype](1e-3),
    tol: Scalar[dtype] = Scalar[dtype](1e-2),
) raises -> Bool:
    # Run forward/backward once to populate analytical gradients
    var logits = model(x)
    var loss = loss_fn(logits, y)
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

            # f(x + eps)
            p.buffer[i] = orig + eps
            var lp = loss_fn(model(x), y).item()

            # f(x - eps)
            p.buffer[i] = orig - eps
            var lm = loss_fn(model(x), y).item()

            # restore
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
# Example test in main
# --------------------
from crossentropy import CrossEntropyLoss
from layers import Linear, ReLU


fn test_gradcheck1() raises:
    var model = Sequential()
    model.append(Linear(4, 5).into())
    model.append(ReLU().into())
    model.append(Linear(5, 3).into())

    var x = Tensor.rand([2, 4], requires_grad=True)
    var y = Tensor.d1([1, 2])

    fn criterion[
        dtype: DType
    ](logits: Tensor[dtype], target: Tensor[dtype]) -> Tensor[dtype]:
        var criterion = CrossEntropyLoss[dtype]()
        return criterion(logits, target)

    var passed = gradcheck_param(
        model,
        x,
        y,
        criterion[DType.float32],
        Scalar[DType.float32](1e-3),
        Scalar[DType.float32](1e-2),
    )
    if passed:
        print(CYAN, "Gradient check PASSED ✅")
    else:
        print(RED, "Gradient check FAILED ❌")

    _ = model


from mse import MSELoss


fn test_gradcheck2() raises:
    var model = Sequential()
    model.append(Linear(4, 5).into())
    model.append(ReLU().into())
    model.append(Linear(5, 3).into())

    var x = Tensor.rand([2, 4], requires_grad=True)
    # Target must match output shape [2, 3]
    var y = Tensor.rand([2, 3])

    fn mse_loss[
        dtype: DType
    ](preds: Tensor[dtype], target: Tensor[dtype]) -> Tensor[dtype]:
        var mse = MSELoss[dtype]()
        return mse(preds, target)

    var passed = gradcheck_param(
        model,
        x,
        y,
        mse_loss[DType.float32],
        Scalar[DType.float32](1e-3),
        Scalar[DType.float32](1e-2),
    )
    if passed:
        print(CYAN, "Gradient check PASSED ✅ (MSE)")
    else:
        print(RED, "Gradient check FAILED ❌ (MSE)")


fn main() raises:
    test_gradcheck2()
