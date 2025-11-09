from tenmo import Tensor
from intlist import IntList
from common_utils import panic
from tiles import TileBackward, Tile


@fieldwise_init
@register_passable
struct Repeat[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        repeat: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        if len(repeat) < self.rank():
            panic(
                "repeat: Number of dimensions of repeat dims ("
                + len(repeat).__str__()
                + ") cannot be smaller than number of dimensions of tensor ("
                + self.rank().__str__()
                + ")"
            )
        return Tile[dtype].forward[track_grad](self, repeat, requires_grad)


fn main() raises:
    pass
