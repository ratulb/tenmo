from tenmo import Tensor
from sys import has_accelerator
from binary_ops_kernel import BinaryOpsKernel


@fieldwise_init
@register_passable
struct BinaryOperation[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        opcode: Int
    ](ref this: Tensor[Self.dtype], ref that: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var out: Tensor[Self.dtype]

        @parameter
        if has_accelerator():
            if this.is_on_gpu() and that.is_on_gpu():
                try:
                    out = BinaryOpsKernel[this.dtype].launch[opcode](this, that)
                except e:
                    print(e)
                    print(
                        "BinaryOperation - GPU operation failed for opcode: ",
                        opcode,
                        ". Failling back on CPU",
                    )
                    out = Tensor[this.dtype](
                        this.buffer.arithmetic_ops[opcode](that.buffer),
                        requires_grad=False,
                    )
            else:
                out = Tensor[this.dtype](
                    this.buffer.arithmetic_ops[opcode](that.buffer),
                    requires_grad=False,
                )
        else:
            out = Tensor[this.dtype](
                this.buffer.arithmetic_ops[opcode](that.buffer),
                requires_grad=False,
            )

        return out^


fn main() raises:
    pass
