from tenmo import Tensor
from tenmo.common_utils import log_debug, RED, CYAN
from tenmo.net import Sequential

comptime dtype = DType.float32

def main() raises:
    test_gradcheck1()
    test_gradcheck2()

# --------------------
# Gradient checker (safe using parameters_ptrs)
# --------------------
def gradcheck_param(
    mut model: Sequential[dtype],
    mut x: Tensor[dtype],
    y: Tensor[DType.int32],
    loss_fn: def(Tensor[dtype], Tensor[DType.int32]) thin -> Tensor[dtype],
    eps: Scalar[dtype] = Scalar[dtype](1e-3),
    tol: Scalar[dtype] = Scalar[dtype](1e-2),
) raises -> Bool where dtype.is_floating_point():
    # Run forward/backward once to populate analytical gradients
    var logits = model(x)
    var loss = loss_fn(logits, y)
    loss.backward()

    var ok = True
    var ptrs = model.parameters()

    for idx in range(len(ptrs)):
        var p_ptr = ptrs[idx]
        var p = p_ptr[]
        log_debug(
            "Gradcheck: checking param len=" + String(len(p.buffer)), CYAN
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
            grad_an = p.gradbox[].buffer()[i]

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
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.net import Linear, ReLU


def test_gradcheck1() raises:
    var model = Sequential[dtype]()
    model.append(Linear[dtype](4, 5).into())
    model.append(ReLU[dtype]().into())
    model.append(Linear[dtype](5, 3).into())

    var x = Tensor[dtype].rand([2, 4], requires_grad=True)
    var y = Tensor[DType.int32].d1([1, 2])

    def criterion(logits: Tensor[dtype], target: Tensor[DType.int32]) -> Tensor[dtype]:
        var criterion = CrossEntropyLoss[dtype]()
        # print("ce logits/target shapes: ", logits.shape, target.shape)
        return criterion(logits, target)

    var passed = gradcheck_param(
        model,
        x,
        y,
        criterion,
        Scalar[DType.float32](1e-3),
        Scalar[DType.float32](1e-2),
    )
    if passed:
        print(CYAN, "Gradient check PASSED ✅")
    else:
        print(RED, "Gradient check FAILED ❌")


from tenmo.mse import MSELoss


def test_gradcheck2() raises:
    var model = Sequential[dtype]()
    model.append(Linear[dtype](4, 5).into())
    model.append(ReLU[dtype]().into())
    model.append(Linear[dtype](5, 3).into())

    var x = Tensor[dtype].rand([2, 4], requires_grad=True)
    # Target must match output shape [2, 3]
    var y = Tensor[DType.int32]([2, 3])

    def mse_loss[
        dtype: DType
    ](preds: Tensor[dtype], target: Tensor[DType.int32]) -> Tensor[dtype]:
        var mse = MSELoss[dtype]()
        return mse(preds, target.to_dtype[dtype]())

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



