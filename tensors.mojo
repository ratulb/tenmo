from memory import UnsafePointer, memcpy
from random import random_ui64, randn, seed
from testing import assert_equal


# Tensor datatype is always float
struct Tensor(Copyable, Movable, Representable, Stringable, Writable):
    var data: UnsafePointer[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Float32].alloc(rows * cols)

    fn __getitem__(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn __setitem__(self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn __moveinit__(out self, owned other: Self):
        self = Self(other.rows, other.cols)
        memcpy(self.data, other.data, self.rows * self.cols)

    fn __copyinit__(out self, other: Self):
        self = Self(other.rows, other.cols)
        memcpy(self.data, other.data, self.rows * self.cols)

    fn __del__(owned self):
        self.data.free()
    
    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn __str__(self) -> String:
        var s = String()
        for i in range(self.rows):
            s += "["
            for j in range(self.cols):
                s += String(self[i, j])
                if j != self.cols - 1:
                    s += ", "
            s += "]"
            if i != self.rows - 1:
                s += "\n"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        s = self.__str__()
        writer.write(s)

    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        seed()
        tensor = Self(rows, cols)
        randn(tensor.data, rows * cols, 0.0, 1.0)
        return tensor

    fn num_elements(self) -> Int:
        return self.rows * self.cols

    fn matmul(self, other: Self) -> Self:
        var result = Tensor(self.rows, other.cols)
        try:
            assert_equal(
                self.cols,
                other.rows,
                "Incompatible shapes for matrix multiplication",
            )
            for i in range(self.rows):
                for j in range(other.cols):
                    var sum: Float32 = 0.0
                    for k in range(self.cols):
                        sum += self[i, k] * other[k, j]
                    result[i, j] = sum
        except e:
            print(e)
        return result

    fn transpose(self) -> Tensor:
        result = Tensor(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        
        return result


fn main():
    this = Tensor(2, 2)
    this[0,0] = 1
    this[0,1] = 2
    this[1,0] = 3
    this[1,1] = 4

    that = Tensor(2, 2)
    that[0,0] = 5
    that[0,1] = 6
    that[1,0] = 7
    that[1,1] = 8
    result = this.matmul(that)
    tome = result
    print(tome)
    transposed = tome.transpose()
    print()
    print(transposed)
    #a = Tensor.rand(4096, 4096)
    #b = Tensor.rand(4096, 4096)
    #c = a.matmul(b)
    #print(c.num_elements())
