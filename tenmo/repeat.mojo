from .tensor import Tensor
from .intarray import IntArray
from .common_utils import panic
from .tiles import TileBackward, Tile


@fieldwise_init
struct Repeat[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        repeat: IntArray,
        requires_grad: Optional[Bool] = None,
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
