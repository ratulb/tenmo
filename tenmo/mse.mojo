from .tensor import Tensor


@fieldwise_init
struct MSELoss[dtype: DType = DType.float32](
    Writable, ImplicitlyCopyable & RegisterPassable
):
    """Mean Squared Error loss.

    Delegates to the mean-reduced `Tensor.mse` op; `training`/`train()`/`eval()`
    switch graph tracking (eval eliminates the autograd graph), matching BCELoss.
    """

    var training: Bool

    def write_to[W: Writer](self, mut writer: W):
        writer.write("MSELoss")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("MSELoss")

    def __init__(out self):
        self.training = True

    def __call__(
        mut self,
        preds: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](preds, target, sync=sync)
        else:
            return Self.forward[track_grad=False](preds, target, sync=sync)

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        preds: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # (1/N) * Σ (preds - target)^2
        return preds.mse[track_grad](target, sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False
