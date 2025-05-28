from common_utils import passthrough
from testing import assert_true


fn test_single_elem_varia_list_creation() raises:
    varia_list = passthrough(5)
    assert_true(
        len(varia_list) == 1,
        "Single element VariadicList length assertion failed",
    )
    assert_true(
        varia_list[0] == 5,
        "Single element VariadicList element assertion failed",
    )
    for e in varia_list:
        assert_true(
            e == 5, "Single element VariadicList iteration assertion failed"
        )


fn main() raises:
    test_single_elem_varia_list_creation()
