from shapes import Shape
from strides import Strides
from buffers import Buffer
from ndbuffer import NDBuffer
from intarray import IntArray
from std.testing import assert_true


fn main() raises:
    # test_explicit_destruction()
    test_buffer_sum_all()


fn test_buffer_sum_all() raises:
    # try:
    print("test_buffer_sum_all")
    comptime dtype = DType.int32
    size = 21
    var l = List[Scalar[dtype]](capacity=size)
    for i in range(size):
        l.append(i)
    var buffer = Buffer[dtype](l)
    var ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    # assert_true(ndb[[2, 6]] == 20)
    assert_true(ndb.sum_all() == 210)
    print("Are we past here 1")
    var shared = ndb.share(Shape(5, 2), offset=1, strides=Strides(2, 2))
    print("********: ", shared.sum(IntArray(), False).item())
    print(shared.sum_all())
    assert_true(shared.sum_all() == 60)
    # assert_true(shared[[4, 1]] == 11)
    # assert_true(ndb[[2, 6]] == 20)
    print("Are we past here 2", ndb.numels(), shared.numels())
    # except e:
    # print(e)
    # raise e^

    _ = """ndb = NDBuffer[dtype](Shape())
    ndb.fill(42)
    assert_true(ndb.sum_all() == 42)
    print("Are we past here 3")
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 42 and shared.item() == 42 and ndb.item() == 42
    )
    print("Are we past here 4")
    # Shape(1)
    ndb = NDBuffer[dtype](Shape(1))
    ndb.fill(39)
    assert_true(ndb.sum_all() == 39 and ndb[IntArray(0)] == 39)
    print("Are we past here 5")
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 39 and shared.item() == 39 and ndb.item() == 39
    )
    print("Are we past here 6")"""
