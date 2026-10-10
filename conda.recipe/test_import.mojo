"""Verify tenmo package is importable and Embedding lookup works."""
from std.testing import assert_true
from tenmo.tensor import Tensor
from tenmo.embedding import Embedding
from tenmo.shared.shapes import Shape


def main() raises:
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=6, embedding_dim=3)
    assert_true(emb.weight.shape() == Shape(6, 3))

    # Lookup rows 2, 3, 5, 1 → one [3]-vector per ID.
    var ids = List[Int]()
    ids.append(2)
    ids.append(3)
    ids.append(5)
    ids.append(1)
    var out = emb(ids)
    assert_true(out.shape() == Shape(4, 3))
    print("tenmo import + embedding lookup OK")
