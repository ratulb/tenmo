from common_utils import single_elem_list
from testing import assert_true
from shapes import Shape


fn test_single_elem_varia_list_creation() raises:
    varia_list = single_elem_list(5)
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


fn test_single_dim_shape_creation() raises:
    shape = Shape.single_dim_shape(5)
    assert_true(
        len(shape.spans()) == 1,
        "Single dim shape VariadicList length assertion failed",
    )
    assert_true(
        shape.spans()[0] == 5,
        "Single dim shape element assertion failed",
    )
    for e in shape.spans():
        assert_true(e == 5, "Single dim shape iteration assertion failed")


fn test_single_dim_shape_creation_with_param(param: Int) raises:
    shape = Shape.single_dim_shape(param)
    assert_true(
        len(shape.spans()) == 1,
        "Single dim shape with param VariadicList length assertion failed",
    )
    assert_true(
        shape.spans()[0] == 5,
        "Single dim with param shape element assertion failed",
    )
    for e in shape.spans():
        assert_true(
            e == 5, "Single dim shape with param iteration assertion failed"
        )


fn test_single_dim_shape_creation_with_param_direct_access_to_axes(
    param: Int,
) raises:
    shape = Shape.single_dim_shape(param)
    assert_true(
        len(shape.axes_spans) == 1,
        (
            "Single dim shape with param axes span VariadicList length"
            " assertion failed"
        ),
    )
    assert_true(
        shape.axes_spans[0] == 5,
        "Single dim with param shape axes span element assertion failed",
    )
    for e in shape.axes_spans:
        assert_true(
            e == 5,
            "Single dim shape with param axes span iteration assertion failed",
        )


fn main() raises:
    test_single_elem_varia_list_creation()
    test_single_dim_shape_creation()
    test_single_dim_shape_creation_with_param(5)
    test_single_dim_shape_creation_with_param_direct_access_to_axes(5)
