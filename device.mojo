from common_utils import panic


struct Device(Equatable, ImplicitlyCopyable, Movable):
    var device: String  # "cpu" or "gpu"
    comptime CPU = Device("cpu")
    comptime GPU = Device("gpu")

    fn __init__(out self, device: String = "cpu"):
        if device == "cpu":
            self.device = "cpu"
        elif device == "gpu":
            self.device = "gpu"
        else:
            self.device = "unknown"
            panic("Invalid device type. Must be 'cpu' or 'gpu'")

    fn __copyinit__(out self, existing: Self):
        self.device = existing.device.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.device = existing.device^

    fn __eq__(self, other: Self) -> Bool:
        return self.device == other.device

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


fn main() raises:
    print("passes")
