from utils import Variant
from operators import __tensor_op_scalar__, MulScalar
from tensors import Tensor
from shapes import Shape
from memory import ArcPointer


@fieldwise_init
struct LongLife[owns_data: Bool = True](Copyable & Movable):
    @parameter
    if owns_data:
        var s: Int



    var refs: Optional[ArcPointer[Int]]

    fn __init__(out self, keep_alive: Bool = False):
        self.refs = Optional(ArcPointer(0)) if keep_alive else None

    fn __del__(var self):
        if self.refs:
            if self.refs.value().count() == 1:
                print("I am going to die")
            else:
                print("Not yet", self.refs.value().count())
        else:
            print("I am dying right away")

    fn keep_living(self):
        print("Still alive")

    fn __moveinit__(out self, owned other: Self):
        self.refs = other.refs

    fn __copyinit__(out self, other: Self):
        # self.refs = other.refs.copy()
        self.refs = other.refs


fn main() raises:
    a = Optional(LongLife(True))
    b = a
    _ = a^
    c = b
    d = c
    b.value().keep_living()
    _ = b^
    c.value().keep_living()
    _ = c
    d.value().keep_living()
    print("Enough of living like alreday dead")

    _ = """shape1 = Shape.Void
    shape2 = Shape.Unit
    print(len(shape1), len(shape2))
    cc = Case("lower")
    ptr = UnsafePointer(to=cc)
    ptr[].s = ptr[].s.upper()
    print(cc.s)
    t = T[DType.float32]()
    result = t.do_it()
    from_inner = result.inner_fn[](100)
    print(from_inner)
    alias Scalars = Variant[Int, String, LoudSpeaker]
    ls = LoudSpeaker()
    v = Scalars(ls)
    v[__type_of(ls)].make_noise()
    v.replace[String, __type_of(ls)]("what is inside now").make_noise()"""


trait Differentiable:
    fn sum[dtype: DType](self, factor: Scalar[dtype]) -> Scalar[dtype]:
        ...


alias GradFn[T: Differentiable & Copyable & Movable] = Variant[T]

alias InnerFn[dtype: DType] = fn (input: Scalar[dtype]) -> Scalar[dtype]


# alias InnerFn = fn (input: String) -> String
@fieldwise_init
struct T[dtype: DType](Copyable & Movable):
    # struct T(Copyable & Movable & ExplicitlyCopyable):
    # var inner_fn: UnsafePointer[Self.InnerFn]
    var inner_fn: Optional[InnerFn[dtype]]

    fn __init__(out self):
        self.inner_fn = None

        _ = """fn __copyinit__(out self, other: Self):
        self.inner_fn = other.inner_fn

    fn __moveinit__(out self, var other: Self):
        self.inner_fn = other.inner_fn

    fn copy(self) -> Self:
        s = Self()
        s.inner_fn = self.inner_fn
        return s"""

    fn do_it(self) -> Self:
        fn some_inner_fn(input: Scalar[dtype]) -> Scalar[dtype]:
            return input * 2

        result = T[dtype]()
        result.inner_fn = Optional(some_inner_fn)
        return result


from utils import Variant


@fieldwise_init
struct Case(Copyable & Movable):
    var s: String


trait Noisy(Copyable & Movable):
    fn make_noise(self):
        ...


@fieldwise_init
struct LoudSpeaker(Copyable & Movable):
    # struct LoudSpeaker(Copyable & Movable & Noisy):
    fn make_noise(self):
        print("Shoutinnnnnnnnnnnnng")
