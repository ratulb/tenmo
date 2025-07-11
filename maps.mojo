from memory import UnsafePointer
from tensors import Tensor
from os import abort


struct TensorMap[dtype: DType = DType.float32](Copyable & Movable):
    var keys: UnsafePointer[Int]
    var values: UnsafePointer[UnsafePointer[Tensor[dtype]]]
    var occupied: UnsafePointer[Scalar[DType.bool]]
    var capacity: Int
    var count: Int

    fn __init__(out self, capacity: Int = 1):
        self.capacity = capacity
        self.count = 0
        self.keys = UnsafePointer[Int].alloc(capacity)
        self.values = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            capacity
        )
        self.occupied = UnsafePointer[Scalar[DType.bool]].alloc(capacity)
        for i in range(capacity):
            self.occupied[i] = False

    fn __moveinit__(out self, owned other: Self):
        self.capacity = other.capacity
        self.count = other.count
        self.keys = other.keys
        self.values = other.values
        self.occupied = other.occupied

    fn __copyinit__(out self, other: Self):
        self.capacity = other.capacity
        self.count = other.count
        self.keys = other.keys
        self.values = other.values
        self.occupied = other.occupied

    fn _hash(self, key: Int) -> Int:
        return key % self.capacity

    fn _resize(mut self) -> None:
        old_keys = self.keys
        old_values = self.values
        old_occupied = self.occupied
        old_capacity = self.capacity

        self.capacity = old_capacity * 2
        self.keys = UnsafePointer[Int].alloc(self.capacity)
        self.values = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            self.capacity
        )
        self.occupied = UnsafePointer[Scalar[DType.bool]].alloc(self.capacity)
        self.count = 0

        for i in range(self.capacity):
            self.occupied[i] = False

        for i in range(old_capacity):
            if old_occupied[i]:
                self.insert(old_keys[i], old_values[i][])

        for i in range(self.capacity):
            (old_keys + i).destroy_pointee()
            (old_values + i).destroy_pointee()
            (old_occupied + i).destroy_pointee()
            old_keys.free()
            old_values.free()
            old_occupied.free()

    #fn insert(mut self, key: Int, value: UnsafePointer[Tensor[dtype]]) -> None:
    fn insert(mut self, key: Int, tensor: Tensor[dtype]) -> None:
        if self.count * 2 >= self.capacity:
            self._resize()

        var idx = self._hash(key)
        var start = idx

        while self.occupied[idx]:
            if self.keys[idx] == key:
                self.values[idx] = tensor.address()
                return
            idx = (idx + 1) % self.capacity
            if idx == start:
                abort("Hash map full even after resize (shouldn't happen)")

        self.keys[idx] = key
        self.values[idx] = tensor.address()
        self.occupied[idx] = True
        self.count += 1

    fn lookup(self, key: Int) -> UnsafePointer[Tensor[dtype]]:
        var idx = self._hash(key)
        var start = idx

        while self.occupied[idx]:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
            if idx == start:
                break
        return UnsafePointer[Tensor[dtype]]()

    fn __contains__(self, key: Int) -> Bool:
        return self.lookup(key).__as_bool__()

    fn __len__(self) -> Int:
        return self.count

    fn __del__(owned self) -> None:
        for i in range(self.capacity):
            (self.keys + i).destroy_pointee()
            (self.values + i).destroy_pointee()
            (self.occupied + i).destroy_pointee()
            self.keys.free()
            self.values.free()
            self.occupied.free()


fn main():
    print("Let's get rolling")
    a = Tensor.d1([1, 2, 3])
    b = Tensor.d1([1, 2, 3, 4])
    map = TensorMap()
    map.insert(1, a.address())
    map.insert(10, b.address())
    result = map.lookup(10)
    result[].print()
    copied = map
    res = copied.lookup(10)
    res[].print()
