def matrix_multiply(A, B, loop_order='ikj'):
    """
    Perform matrix multiplication with specified loop ordering.
    Args:
        A (list of lists): First matrix (m×n)
        B (list of lists): Second matrix (n×p)
        loop_order (str): Order of loops ('ijk', 'ikj', 'jik', 'jki', 'kij', 'kji')
    Returns:
        list of lists: Resultant matrix (m×p)
    """
    # Input validation
    if not (isinstance(A, list) and all(isinstance(row, list) for row in A)):
        raise ValueError("A must be a list of lists")
    if not (isinstance(B, list) and all(isinstance(row, list) for row in B)):
        raise ValueError("B must be a list of lists")
    if not A or not B:
        raise ValueError("Matrices cannot be empty")
    m = len(A)
    n_A = len(A[0]) if m > 0 else 0
    n_B = len(B)
    p = len(B[0]) if n_B > 0 else 0
    # Check if all rows in A have same length
    for row in A:
        if len(row) != n_A:
            raise ValueError("All rows in A must have the same length")
    # Check if all rows in B have same length
    for row in B:
        if len(row) != p:
            raise ValueError("All rows in B must have the same length")
    # Check if matrices can be multiplied
    if n_A != n_B:
        raise ValueError(f"Incompatible dimensions: A is {m}x{n_A}, B is {n_B}x{p}")
    # Initialize result matrix with zeros
    C = [[0 for _ in range(p)] for _ in range(m)]
    # Perform multiplication based on loop order
    if loop_order == 'ijk':
        # i-j-k order (classic)
        for i in range(m):          # rows of A and C
            for j in range(p):      # columns of B and C
                for k in range(n_A): # columns of A and rows of B
                    C[i][j] += A[i][k] * B[k][j]
    elif loop_order == 'ikj':
        # i-k-j order (often better cache performance)
        for i in range(m):          # rows of A and C
            for k in range(n_A):     # columns of A and rows of B
                for j in range(p):  # columns of B and C
                    C[i][j] += A[i][k] * B[k][j]
    elif loop_order == 'jik':
        # j-i-k order
        for j in range(p):          # columns of B and C
            for i in range(m):       # rows of A and C
                for k in range(n_A): # columns of A and rows of B
                    C[i][j] += A[i][k] * B[k][j]
    elif loop_order == 'jki':
        # j-k-i order
        for j in range(p):          # columns of B and C
            for k in range(n_A):     # columns of A and rows of B
                for i in range(m):   # rows of A and C
                    C[i][j] += A[i][k] * B[k][j]
    elif loop_order == 'kij':
        # k-i-j order
        for k in range(n_A):         # columns of A and rows of B
            for i in range(m):       # rows of A and C
                for j in range(p):   # columns of B and C
                    C[i][j] += A[i][k] * B[k][j]
    elif loop_order == 'kji':
        # k-j-i order
        for k in range(n_A):         # columns of A and rows of B
            for j in range(p):       # columns of B and C
                for i in range(m):   # rows of A and C
                    C[i][j] += A[i][k] * B[k][j]
    else:
        raise ValueError("Invalid loop order. Must be one of: 'ijk', 'ikj', 'jik', 'jki', 'kij', 'kji'")
    return C


def test_matrix_multiplication():
    # Test matrices
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    B = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
    # Expected result
    expected = [
        [58, 64],
        [139, 154]
    ]
    # Test all loop orders
    for order in ['ijk', 'ikj', 'jik', 'jki', 'kij', 'kji']:
        result = matrix_multiply(A, B, loop_order=order)
        assert result == expected, f"Failed for order {order}"
        print(f"Order {order} passed: {result}")
    # Test input validation
    try:
        matrix_multiply([[1, 2], [3]], [[1, 2], [3, 4]])  # Invalid A
    except ValueError as e:
        print(f"Caught invalid A: {e}")
    try:
        matrix_multiply([[1, 2], [3, 4]], [[1, 2], [3]])  # Invalid B
    except ValueError as e:
        print(f"Caught invalid B: {e}")
    try:
        matrix_multiply([[1, 2]], [[1], [2], [3]])  # Incompatible dimensions
    except ValueError as e:
        print(f"Caught incompatible dimensions: {e}")
    print("All tests passed!")

if __name__ == "__main__":
    test_matrix_multiplication()
