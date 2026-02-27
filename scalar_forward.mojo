from tenmo import Tensor
from sys import has_accelerator
from scalar_ops_kernel import ScalarOpsKernel


@fieldwise_init
@register_passable
struct ScalarOperation[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        opcode: Int
    ](ref self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out: Tensor[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = ScalarOpsKernel[Self.dtype].launch[opcode](
                        self, scalar
                    )
                except e:
                    print(e)
                    print(
                        "ScalarOperation - GPU operation failed for opcode: ",
                        opcode,
                        ". Failling back on CPU",
                    )
                    out = Tensor[Self.dtype](
                        self.buffer.scalar_ops[opcode](scalar),
                        requires_grad=False,
                    )

            else:
                out = Tensor[Self.dtype](
                    self.buffer.scalar_ops[opcode](scalar),
                    requires_grad=False,
                )

        else:
            out = Tensor[Self.dtype](
                self.buffer.scalar_ops[opcode](scalar), requires_grad=False
            )

        return out^


fn main() raises:
    pass
