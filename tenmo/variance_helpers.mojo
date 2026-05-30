from .ndbuffer import NDBuffer
from .buffers import Buffer
from .device import DeviceState
from .common_utils import panic
from .shapes import Shape
from std.sys.info import has_accelerator
from .std_variance_backward_kernel import StdVarianceBackwardKernel


@fieldwise_init
struct VarStdBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def variance_backward_normalize(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        scale: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused variance backward: (x - mean) * scale.

        Single pass over (*, D).
        x may be non-contiguous — stride-aware on CPU and GPU.
        out is always a fresh contiguous allocation.
        """
        comptime if has_accelerator():
            if x.is_on_gpu():
                try:
                    return StdVarianceBackwardKernel[
                        Self.dtype
                    ].launch_variance_backward(x, mean, scale)
                except e:
                    print(e)
                    panic(
                        "VarStdBackward variance_backward_normalize → GPU"
                        " failed"
                    )
                    return NDBuffer[Self.dtype].Empty()
        return Self._variance_backward_normalize_cpu(x, mean, scale)

    @staticmethod
    def _variance_backward_normalize_cpu(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        scale: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """CPU variance backward normalize — stride-aware via x[coord]."""
        var D = x.shape[-1]
        var out_buf = Buffer[Self.dtype](x.numels())
        var out_shape = x.shape

        var row = 0
        for outer_coord in x.shape[0:-1]:
            var row_mean = mean.buffer[row]
            var out_base = row * D
            for i in range(D):
                var full_coord = outer_coord.insert(len(outer_coord), i)
                out_buf[out_base + i] = (x[full_coord] - row_mean) * scale
            row += 1

        return NDBuffer[Self.dtype](out_buf^, out_shape)

    @staticmethod
    def std_backward_normalize(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        denom: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused std backward: (x - mean) / denom.

        Single pass over (*, D).
        x may be non-contiguous — stride-aware on CPU and GPU.
        out is always a fresh contiguous allocation.
        """
        comptime if has_accelerator():
            if x.is_on_gpu():
                try:
                    return StdVarianceBackwardKernel[
                        Self.dtype
                    ].launch_std_backward(x, mean, denom)
                except e:
                    print(e)
                    panic("VarStdBackward std_backward_normalize → GPU failed")
                    return NDBuffer[Self.dtype].Empty()
        return Self._std_backward_normalize_cpu(x, mean, denom)

    @staticmethod
    def _std_backward_normalize_cpu(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        denom: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """CPU std backward normalize — stride-aware via x[coord]."""
        var D = x.shape[-1]
        var out_buf = Buffer[Self.dtype](x.numels())
        var out_shape = x.shape

        var row = 0
        for outer_coord in x.shape[0:-1]:
            var row_mean = mean.buffer[row]
            var row_denom = denom.buffer[row]
            var out_base = row * D
            for i in range(D):
                var full_coord = outer_coord.insert(len(outer_coord), i)
                out_buf[out_base + i] = ((x[full_coord] - row_mean)) / row_denom
            row += 1

        return NDBuffer[Self.dtype](out_buf^, out_shape)
