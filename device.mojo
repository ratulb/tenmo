from common_utils import panic
from gpu.host import DeviceContext
from memory import ArcPointer
from sys import has_accelerator
from utils import Variant

comptime DeviceType = Variant[CPU, GPU]


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, Movable):
    var kind: DeviceType

    fn __init__(out self):
        self.kind = CPU()


@fieldwise_init
struct CPU(Equatable, ImplicitlyCopyable, Movable):
    fn __eq__(self, other: Self) -> Bool:
        return True

    fn __ne__(self, other: Self) -> Bool:
        return False


struct GPU(Equatable, ImplicitlyCopyable, Movable):
    var device_context: ArcPointer[DeviceContext]

    fn into(self) -> Device:
        return Device(self)

    fn __init__(out self, device_id: Int = 0) raises:
        self.device_context = ArcPointer(DeviceContext(device_id))

    fn __copyinit__(out self, existing: Self):
        self.device_context = existing.device_context.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.device_context = existing.device_context^

    fn __eq__(self, other: Self) -> Bool:
        return self.device_context.__is__(other.device_context)

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __enter__(mut self) -> Self:
        return self.copy()

    fn __exit__(mut self):
        print("Context exit")

    fn __getitem__(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn handle(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn __call__(self) -> DeviceContext:
        return self.device_context.copy()[]


fn main() raises:
    var device = Device()
    # var device = Device(CPU())
    print("passes")
