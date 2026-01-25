from gpu.host import DeviceContext
from buffers import Buffer

comptime HEIGHT = 2
comptime WIDTH = 3
comptime dtype = DType.float32


fn kernel[dtype: DType](buffer: Buffer[dtype]):
    print(buffer)
    buffer.__iadd__(10)
    print(buffer)


def main():
    ctx = DeviceContext()
    a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
    a.enqueue_fill(42)

    buffer = Buffer[dtype](HEIGHT * WIDTH, a)
    ctx.enqueue_function[kernel[dtype], kernel[dtype]](
        buffer, grid_dim=1, block_dim=1
    )
    ctx.synchronize()

    with a.map_to_host() as out_buf_host:
        print("out:", out_buf_host)
