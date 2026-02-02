from gpu import thread_idx
from gpu.host import DeviceContext

comptime SIZE = 4
comptime MATRIX_SIZE = 3
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32


fn process_sliding_window[
    datatype: DType = DType.float32
](
    output: UnsafePointer[Scalar[datatype], MutAnyOrigin],
    a: UnsafePointer[Scalar[datatype], ImmutAnyOrigin],
):
    thread_id = Int(thread_idx.x)

    # Each thread processes a sliding window of 3 elements
    window_sum = Scalar[datatype](0.0)
    size = SIZE

    var (left, centre, right) = thread_id - 1, thread_id, thread_id + 1

    if thread_id < size:
        if left > 0:
            window_sum += a[left]
        if right < size:
            window_sum += a[right]
        window_sum += a[centre]
        output[centre] = window_sum


fn main() raises:
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        # Initialize input [0, 1, 2, 3]
        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i

        ctx.enqueue_function[
            process_sliding_window[dtype], process_sliding_window[dtype]
        ](
            output_buf,
            input_buf,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with output_buf.map_to_host() as output_host:
            expected_0 = Scalar[dtype](1.0)
            expected_1 = Scalar[dtype](3.0)
            expected_2 = Scalar[dtype](6.0)
            expected_3 = Scalar[dtype](5.0)
            matches = True
            if abs(output_host[0] - expected_0) > 0.001:
                matches = False
            if abs(output_host[1] - expected_1) > 0.001:
                matches = False
            if abs(output_host[2] - expected_2) > 0.001:
                matches = False
            if abs(output_host[3] - expected_3) > 0.001:
                matches = False

            if matches:
                print("[PASS] Test PASSED - Sliding window sums are correct")
            else:
                print("[FAIL] Test FAILED - Sliding window sums are incorrect!")
