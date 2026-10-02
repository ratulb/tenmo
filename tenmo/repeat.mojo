from .tensor import Tensor
from .shared.intarray import IntArray
from .shared.panic import panic
from .tiles import Tile


@fieldwise_init
struct Repeat[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        repeat: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if len(repeat) < self.rank():
            panic(
                "repeat: Number of dimensions of repeat dims ("
                + String(len(repeat))
                + ") cannot be smaller than number of dimensions of tensor ("
                + String(self.rank())
                + ")"
            )
        return Tile[Self.dtype].forward[track_grad](self, repeat, requires_grad)
