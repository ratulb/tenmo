from tenmo import Tensor, Shape
from tenmo.common_utils import i, s
from tenmo.mnemonics import dot

comptime dtype = DType.float32

def main() raises:

    var data = Tensor[dtype].d2(
        [[0.43, 0.15, 0.89],
        #[0.55, 0.87, 0.66],
        #[0.57, 0.85, 0.64],
        #[0.22, 0.58, 0.33],
        #[0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]]
    )
    _="""var query = data[i(1), s()]
    var attn_scores_2 = Tensor[dtype].empty(Shape(data.shape()[0]))

    for i, x_i in enumerate(data.slices()):
        attn_scores_2[i] = query.dot(x_i).item()

    print("attn_scores_2:\n")
    attn_scores_2.print()

    res: Float32 = 0
    for idx, element in enumerate(data[i(0), s()].slices()):
        res += element.item() * query[idx]
    print(res)
    print(data[i(0), s()].dot(query).item())

    var attn_weights_2 = attn_scores_2.softmax(axes=[0])
    print("attn_weights_2: \n")
    attn_weights_2.print()
    print("attn_weights_2 sum: \n")
    attn_weights_2.sum().print()
    var context_vec_2 = Tensor[dtype].zeros(query.shape())
    for i,x_i in enumerate(data.slices()):
        context_vec_2 += attn_weights_2[i]*x_i
    print("context_vec_2: \n")
    context_vec_2.print()"""


    _="""attn_scores = Tensor[dtype].empty(6, 6)
    for i, x_i in enumerate(data.slices()):
        for j, x_j in enumerate(data.slices()):
            attn_scores[i, j] = Tensor[dtype].matmul[mode=dot](x_i, x_j).item()
    print("attn_scores: \n")
    attn_scores.print()"""

    attn_scores = data.matmul(data.transpose())
    print("attn_scores oneshot\n")
    attn_scores.print()

    attn_weights = attn_scores.softmax(axes=[-1])
    print("attn_weights\n")
    attn_weights.print()

    attn_weights.matmul(data).print()
