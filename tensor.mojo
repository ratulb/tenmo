from memory import UnsafePointer, memcpy, memset


struct Tensor[dtype: DType = DType.float32]:
    var data: UnsafePointer[Scalar[dtype]]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Scalar[dtype]].alloc(rows * cols)

    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        return self.data[row * self.cols + col]

    fn __setitem__(self, row: Int, col: Int, value: Scalar[dtype]):
        self.data[row * self.cols + col] = value

    fn __moveinit__(out self, owned other: Self):
        self = Self(other.rows, other.cols)
        other.data.move_pointee_into(self.data)

    fn __copyinit__(out self, other: Self):
        self = Self(other.rows, other.cols)
        memcpy(self.data, other.data, self.rows * self.cols)

    fn __del__(owned self):
        self.data.free()

    fn data_type(self) -> DType:
        return dtype

    fn num_elements(self) -> Int:
        return self.rows * self.cols


fn main():
    print("In main")
    tensor1 = Tensor[DType.uint8](10, 10)
    print(tensor1.data_type())
    tensor1[0, 0] = 200
    print(tensor1[0, 0])
    tensor2 = Tensor(10, 10)
    print(tensor2.data_type())
    print(tensor2[0, 0])
