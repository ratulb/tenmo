from tenmo import Tensor
from matmul_kernel import MatmulNdGpu
from testing import assert_true
from sys import has_accelerator
from intarray import IntArray
from shapes import Shape

comptime dtype = DType.float32


fn test_gpu_transfer_fidelity() raises:
    print("=== Test 1: GPU transfer fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var B_back = B_gpu.to_cpu()
    assert_true(B.all_close(B_back))
    print("PASSED: B == B_gpu.to_cpu()")


fn test_ancestry_storage_fidelity() raises:
    print("=== Test 2: Ancestry storage fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var A_gpu = A.to_gpu()
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)
    var B_from_ancestry = C_gpu.ancestry().get(1)
    var B_ancestry_back = B_from_ancestry.to_cpu()
    assert_true(B.all_close(B_ancestry_back))
    print("PASSED: B from ancestry == original B")


fn test_forward_matmul_fidelity() raises:
    print("=== Test 3: Forward matmul fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)
    assert_true(C_cpu.all_close(C_gpu.to_cpu()))
    print("PASSED: CPU matmul == GPU matmul")


fn test_backward_grad_A_fidelity_orig() raises:
    print("=== Test 4: Backward grad_A fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)

    C_cpu.backward()
    var A_cpu_grad = A.grad().copy()
    A.zero_grad()

    # print("A_gpu grad before backward (should be all zeros):")
    # A_gpu.grad().print()

    assert_true(A_gpu.grad().all_close(Tensor[dtype].zeros(Shape(9, 80))))

    C_gpu.backward()

    _ = """print("A_gpu grad after backward:")
    A_gpu.grad().print()
    print("A CPU grad:")
    A_cpu_grad.print()"""
    print("Grad calculated on CPU")
    A_cpu_grad.print()
    assert_true(A_gpu.grad().all_close(A_cpu_grad), "Direct extraction failed")

    assert_true(A.grad().all_close(A_cpu_grad))
    print("PASSED: GPU backward grad_A == CPU backward grad_A")


fn test_backward_grad_A_fidelity() raises:
    print("=== Test 4: Backward grad_A fidelity ===")
    var A = Tensor[dtype].arange(9 * 30)
    var B = Tensor[dtype].arange(30 * 5)

    A = A.reshape(Shape(9, 30), requires_grad=True)
    B = B.reshape(Shape(30, 5))

    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)

    C_cpu.backward()
    var A_cpu_grad = A.grad().copy()
    A.zero_grad()

    # print("A_gpu grad before backward (should be all zeros):")
    # A_gpu.grad().print()

    assert_true(A_gpu.grad().all_close(Tensor[dtype].zeros(Shape(9, 30))))

    C_gpu.backward()

    _ = """print("A_gpu grad after backward:")
    A_gpu.grad().print()
    print("A CPU grad:")
    A_cpu_grad.print()"""
    print("Grad calculated on CPU")
    A_cpu_grad.print()
    assert_true(A_gpu.grad().all_close(A_cpu_grad), "Direct extraction failed")

    assert_true(A.grad().all_close(A_cpu_grad))
    print("PASSED: GPU backward grad_A == CPU backward grad_A")


fn test_transposed_matmul_fidelity() raises:
    print("=== Test 5: Transposed matmul fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var BT_gpu = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu
    )
    var grad_A_GPU = Tensor[dtype](grad_A_ndb^)
    var grad_A_gpu = grad_A_GPU.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A row 0:")
    for i in range(min(8, grad_A_gpu.shape()[-1])):
        print(grad_A_gpu.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_gpu))
    print("PASSED: GPU transposed matmul == CPU")


fn test_ancestry_transposed_matmul_fidelity() raises:
    print("=== Test 6: B from ancestry transposed matmul ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var B_anc = C_gpu.ancestry().get(1)
    var BT_anc = B_anc.buffer.transpose(axes=IntArray(-1, -2))

    var grad_A_anc_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_anc
    )
    var grad_A_ANC = Tensor[dtype](grad_A_anc_ndb^)
    var grad_A_anc = grad_A_ANC.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A from ancestry row 0:")
    for i in range(min(8, grad_A_anc.shape()[-1])):
        print(grad_A_anc.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_anc))
    print("PASSED: B from ancestry transposed matmul == CPU")

from common_utils import now
fn main() raises:
    _="""comptime dtype = DType.float32
    print("=== Test 4: Backward grad_A fidelity ===")
    var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
    A = A.reshape(Shape(9, 30), requires_grad=True)
    A = A.contiguous()
    var B = Tensor[dtype].arange(30 * 5)
    B = B.reshape(Shape(30, 5))
    var runs = 1000000
    var start = now()
    var C = A.matmul(B)
    for _ in range(runs):
        A.zero_grad()
        C = A.matmul(B)
        #print("here we come *********")
        C.backward()

    print("Time 1: ", (now() - start) * 1000, "ms")
    A.grad().print()
    var A_like = Tensor[dtype].ones_like(C)
    var B_trans = B.transpose()
    start = now()
    for _ in range(runs):
        _ = A_like.buffer.matmul_2d[tile_size=32](B_trans.buffer)


    (A_like.buffer.matmul_2d[tile_size=32](B_trans.buffer)).print()
    print("Time 2: ", (now() - start) * 1000, "ms")
    #prints
    # [2D Gradbox(9, 30), Type: float32, Shared : False, Strides : (30, 1), Offset : 0, Device : cpu]
    #[
    #[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, ..., 510.0, 535.0, 560.0, 585.0, 610.0, 635.0, 660.0, 685.0, 710.0, 735.0],
    #....
    #....
    # PyTorch print tensor([[ 10.,  35.,  60.,  85., 110., 135., 160., 185., 210., 235., 260., 285.,.....
    # GPU implementation prints [10.0, 35.0, 60.0, 85.0, 110.0, 135.0, 160.0, 185.0, 210.0, 235.0, ..., 510.0, 535.0, 560.0, 585.0, 610.0, 635.0, 660.0, 685.0, 710.0, 735.0],..."""
    @parameter
    if not has_accelerator():
        print("No GPU available — skipping tests")
        return
    else:
        test_gpu_transfer_fidelity()
        test_ancestry_storage_fidelity()
        test_forward_matmul_fidelity()
        test_ancestry_transposed_matmul_fidelity()
        test_transposed_matmul_fidelity()
        test_backward_grad_A_fidelity()

        print("\n=== ALL TESTS PASSED ===")


fn get_A() -> Tensor[DType.float32]:
    return Tensor[DType.float32].d2(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
            ],
            [
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                36.0,
                37.0,
                38.0,
                39.0,
                40.0,
                41.0,
                42.0,
                43.0,
                44.0,
                45.0,
                46.0,
                47.0,
                48.0,
                49.0,
                50.0,
                51.0,
                52.0,
                53.0,
                54.0,
                55.0,
                56.0,
                57.0,
                58.0,
                59.0,
            ],
            [
                60.0,
                61.0,
                62.0,
                63.0,
                64.0,
                65.0,
                66.0,
                67.0,
                68.0,
                69.0,
                70.0,
                71.0,
                72.0,
                73.0,
                74.0,
                75.0,
                76.0,
                77.0,
                78.0,
                79.0,
                80.0,
                81.0,
                82.0,
                83.0,
                84.0,
                85.0,
                86.0,
                87.0,
                88.0,
                89.0,
            ],
            [
                90.0,
                91.0,
                92.0,
                93.0,
                94.0,
                95.0,
                96.0,
                97.0,
                98.0,
                99.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                119.0,
            ],
            [
                120.0,
                121.0,
                122.0,
                123.0,
                124.0,
                125.0,
                126.0,
                127.0,
                128.0,
                129.0,
                130.0,
                131.0,
                132.0,
                133.0,
                134.0,
                135.0,
                136.0,
                137.0,
                138.0,
                139.0,
                140.0,
                141.0,
                142.0,
                143.0,
                144.0,
                145.0,
                146.0,
                147.0,
                148.0,
                149.0,
            ],
            [
                150.0,
                151.0,
                152.0,
                153.0,
                154.0,
                155.0,
                156.0,
                157.0,
                158.0,
                159.0,
                160.0,
                161.0,
                162.0,
                163.0,
                164.0,
                165.0,
                166.0,
                167.0,
                168.0,
                169.0,
                170.0,
                171.0,
                172.0,
                173.0,
                174.0,
                175.0,
                176.0,
                177.0,
                178.0,
                179.0,
            ],
            [
                180.0,
                181.0,
                182.0,
                183.0,
                184.0,
                185.0,
                186.0,
                187.0,
                188.0,
                189.0,
                190.0,
                191.0,
                192.0,
                193.0,
                194.0,
                195.0,
                196.0,
                197.0,
                198.0,
                199.0,
                200.0,
                201.0,
                202.0,
                203.0,
                204.0,
                205.0,
                206.0,
                207.0,
                208.0,
                209.0,
            ],
            [
                210.0,
                211.0,
                212.0,
                213.0,
                214.0,
                215.0,
                216.0,
                217.0,
                218.0,
                219.0,
                220.0,
                221.0,
                222.0,
                223.0,
                224.0,
                225.0,
                226.0,
                227.0,
                228.0,
                229.0,
                230.0,
                231.0,
                232.0,
                233.0,
                234.0,
                235.0,
                236.0,
                237.0,
                238.0,
                239.0,
            ],
            [
                240.0,
                241.0,
                242.0,
                243.0,
                244.0,
                245.0,
                246.0,
                247.0,
                248.0,
                249.0,
                250.0,
                251.0,
                252.0,
                253.0,
                254.0,
                255.0,
                256.0,
                257.0,
                258.0,
                259.0,
                260.0,
                261.0,
                262.0,
                263.0,
                264.0,
                265.0,
                266.0,
                267.0,
                268.0,
                269.0,
            ],
        ],
        requires_grad=True,
    )


fn get_B() -> Tensor[DType.float32]:
    return Tensor[DType.float32].d2(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0],
            [30.0, 31.0, 32.0, 33.0, 34.0],
            [35.0, 36.0, 37.0, 38.0, 39.0],
            [40.0, 41.0, 42.0, 43.0, 44.0],
            [45.0, 46.0, 47.0, 48.0, 49.0],
            [50.0, 51.0, 52.0, 53.0, 54.0],
            [55.0, 56.0, 57.0, 58.0, 59.0],
            [60.0, 61.0, 62.0, 63.0, 64.0],
            [65.0, 66.0, 67.0, 68.0, 69.0],
            [70.0, 71.0, 72.0, 73.0, 74.0],
            [75.0, 76.0, 77.0, 78.0, 79.0],
            [80.0, 81.0, 82.0, 83.0, 84.0],
            [85.0, 86.0, 87.0, 88.0, 89.0],
            [90.0, 91.0, 92.0, 93.0, 94.0],
            [95.0, 96.0, 97.0, 98.0, 99.0],
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [105.0, 106.0, 107.0, 108.0, 109.0],
            [110.0, 111.0, 112.0, 113.0, 114.0],
            [115.0, 116.0, 117.0, 118.0, 119.0],
            [120.0, 121.0, 122.0, 123.0, 124.0],
            [125.0, 126.0, 127.0, 128.0, 129.0],
            [130.0, 131.0, 132.0, 133.0, 134.0],
            [135.0, 136.0, 137.0, 138.0, 139.0],
            [140.0, 141.0, 142.0, 143.0, 144.0],
            [145.0, 146.0, 147.0, 148.0, 149.0],
        ]
    )
