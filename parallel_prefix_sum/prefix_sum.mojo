from std.testing import assert_true, TestSuite

def cpu_scan(inputs: List[Int]) -> List[Int]:
    """Sequential prefix sum (exclusive)."""
    var output = List[Int](capacity=len(inputs))
    var sum = 0
    for i in range(len(inputs)):
        output.append(sum)
        sum += inputs[i]
    return output^


def validate_scan(original: List[Int], scanned: List[Int]) raises:
    # Compare with CPU scan
    var expected = cpu_scan(original)
    assert_true(expected == scanned)

    _ = """# Implement naive scan kernel (single block, shared memory)
    fn naive_scan_kernel[
        dtype: DType = DType.float32
    ](
        output: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
        input: UnsafePointer[Scalar[dtype], ImmutUnsafeAnyOrigin],
        n: Int
    ):
        #""Naive parallel scan - O(n log n) operations""
        # Allocate shared memory
        var temp = gpu_ctx.enqueue_alloc_shared_memory[dtype](n)

        # Load input into shared memory (exclusive scan: shift right)
        var tid = thread_idx.x
        var global_idx = block_idx.x * block_dim.x + tid

        if tid == 0:
            temp[0] = 0  # Identity element
        else:
            temp[tid] = input[global_idx - 1]

        gpu_ctx.synchronize()

        # Scan loop
        var offset = 1
        var pout = 0
        var pin = 1

        while offset < n:
            if tid >= offset:
                var val = temp[pin * n + tid - offset]
                temp[pout * n + tid] += val
            else:
                temp[pout * n + tid] = temp[pin * n + tid]

            # Swap buffers
            pout, pin = pin, pout
            gpu_ctx.synchronize()
            offset *= 2

        # Write output
        output[global_idx] = temp[pout * n + tid]"""


def main() raises:
    validate_scan([3, 1, 7, 0, 4, 1, 6, 3], [0, 3, 4, 11, 11, 15, 16, 22])
