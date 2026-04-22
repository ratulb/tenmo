from .shapes import Shape
from .broadcasthelper import ShapeBroadcaster
from .common_utils import panic


fn main():
    pass


@fieldwise_init
struct MatrixShapeValidator(ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn validate_matrix_shapes_nd(A_shape: Shape, B_shape: Shape):
        if len(A_shape) < 2 or len(B_shape) < 2:
            panic(
                "MatrixShapeValidator → validate_matrix_shapes_nd: matmul_nd"
                " expects rank >= 2. Got A = "
                + String(A_shape)
                + ", B = "
                + String(B_shape)
            )

        if A_shape[-1] != B_shape[-2]:
            panic(
                "MatrixShapeValidator → validate_matrix_shapes_nd: inner"
                " dimensions mismatch: "
                + "A(...,"
                + String(A_shape[-1])
                + ") vs "
                + "B("
                + String(B_shape[-2])
                + ",...). "
                + "Full A = "
                + String(A_shape)
                + ", B = "
                + String(B_shape)
            )

        A_batch = A_shape[0:-2]
        B_batch = B_shape[0:-2]
        _ = ShapeBroadcaster.broadcast_shape(
            A_batch, B_batch
        )  # will panic internally if not compatible

    @always_inline
    @staticmethod
    fn validate_matrix_shapes_2d(A_shape: Shape, B_shape: Shape):
        if len(A_shape) != 2 or len(B_shape) != 2:
            panic(
                "MatrixShapeValidator → validate_matrix_shapes_2d: matmul_2d"
                " expects rank == 2. Got A = "
                + String(A_shape)
                + ", B = "
                + String(B_shape)
            )

        if A_shape[1] != B_shape[0]:
            panic(
                "MatrixShapeValidator → validate_matrix_shapes_2d: inner"
                " dimensions mismatch in matmul_2d: "
                + "A(m,"
                + String(A_shape[1])
                + ") vs "
                + "B("
                + String(B_shape[0])
                + ",n). "
                + "Full A = "
                + String(A_shape)
                + ", B = "
                + String(B_shape)
            )
