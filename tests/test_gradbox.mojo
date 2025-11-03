from buffers import Buffer
from testing import assert_true
from ndbuffer import NDBuffer
from shapes import Shape
from gradbox import Gradbox

fn main() raises:
    run = 1
    for _ in range(run):
        test_gradbox_is_shared()
        test_seed_gradbox()
        test_gradbox_inplace_add()
        test_gradbox_reshape()
        test_gradbox_reverse_subtract()
        test_gradbox_reverse_division()


fn test_gradbox_reverse_division() raises:
    print("test_gradbox_reverse_division")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 / gradbox
    assert_true(
        result.buffer.data()
        == Buffer[dtype]([2.0, 1.0, 0.6666667, 0.5, 0.4, 0.33333334])
    )


fn test_gradbox_reverse_subtract() raises:
    print("test_gradbox_reverse_subtract")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 - gradbox
    assert_true(result.buffer.data() == Buffer[dtype]([1, 0, -1, -2, -3, -4]))


fn test_gradbox_reshape() raises:
    print("test_gradbox_reshape")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    reshaped = gradbox.reshape(Shape(3, 2))
    assert_true(reshaped[[2, 1]] == 6 and reshaped[[1, 1]] == 4)
    reshaped.zero_grad()
    assert_true(reshaped[[2, 1]] == 0 and reshaped[[1, 1]] == 0)
    assert_true(gradbox[[1, 2]] == 6 and gradbox[[0, 1]] == 2)


fn test_gradbox_inplace_add() raises:
    print("test_gradbox_inplace_add")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)

    buffer2 = Buffer[dtype]([11, 12, 13, 14, 15, 16])
    ndb2 = NDBuffer[dtype](buffer2^, Shape(2, 3))
    gradbox2 = Gradbox[dtype](ndb2^)

    gradbox += gradbox2
    assert_true(
        gradbox.buffer.data() == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(gradbox.buffer.buffer == None)


fn test_gradbox_is_shared() raises:
    print("test_gradbox_is_shared")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared(), "Gradbox buffer is shared - assertion failed"
    )


fn test_seed_gradbox() raises:
    print("test_seed_gradbox")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([1, 2, 3, 4, 5, 6])
    )
    gradbox.seed_grad(42)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([42, 42, 42, 42, 42, 42])
    )
