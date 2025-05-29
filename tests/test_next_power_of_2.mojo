from common_utils import next_power_of_2
from testing import assert_true


fn test_next_power_of_2() raises:
    assert_true(
        next_power_of_2(1) == 1,
        "Next power of 2 assertion failed",
    )

    assert_true(
        next_power_of_2(2) == 2,
        "Next power of 2 assertion failed",
    )
    assert_true(
        next_power_of_2(3) == 4,
        "Next power of 2 assertion failed",
    )
    assert_true(
        next_power_of_2(4) == 4,
        "Next power of 2 assertion failed",
    )

    assert_true(
        next_power_of_2(5) == 8,
        "Next power of 2 assertion failed",
    )
    assert_true(
        next_power_of_2(6) == 8,
        "Next power of 2 assertion failed",
    )
    assert_true(
        next_power_of_2(7) == 8,
        "Next power of 2 assertion failed",
    )
    assert_true(
        next_power_of_2(8) == 8,
        "Next power of 2 assertion failed",
    )


fn main() raises:
    test_next_power_of_2()
