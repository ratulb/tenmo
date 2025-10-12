from tensors import Tensor


@explicit_destroy
struct SGD[dtype: DType = DType.float32](Copyable & Movable):
    """
    Stochastic Gradient Descent (SGD) Optimizer.

    This struct implements a minimal version of the SGD optimizer.
    It directly mutates model parameters in place using unsafe pointers
    to avoid value-semantic copies (Mojo’s default).

    ⚠️ Important:
    - Parameters must outlive the optimizer, since we store raw
      `UnsafePointer`s to them.
    - Updates are applied directly to `.buffer` (like PyTorch’s `.data`)
      to avoid "leaf tensor in-place" autograd restrictions.
    """

    # List of raw pointers to parameters that should be updated.
    var params: List[UnsafePointer[Tensor[dtype]]]
    # Learning rate (step size for updates).
    var lr: Scalar[dtype]
    # Whether to automatically zero grads after each update.
    var zero_grad_post_step: Bool

    """
    Initialize an SGD optimizer.

    Arguments:
      params: list of UnsafePointer[Tensor], typically produced with `addr()` or `addrs()`.
      lr: learning rate (default = 0.01).
      zero_grad_post_step: if True, clears gradients after each step.
    """

    fn __init__(
        out self,
        params: List[UnsafePointer[Tensor[dtype]]],
        lr: Scalar[dtype] = Scalar[dtype](0.01),
        zero_grad_post_step: Bool = True,
    ):
        self.params = params
        self.lr = lr
        self.zero_grad_post_step = zero_grad_post_step

    # Copy/move initializers (default shallow pointer copies are fine).
    fn __copyinit__(out self, existing: Self):
        self.params = existing.params
        self.lr = existing.lr
        self.zero_grad_post_step = existing.zero_grad_post_step

    fn __moveinit__(out self, deinit existing: Self):
        self.params = existing.params
        self.lr = existing.lr
        self.zero_grad_post_step = existing.zero_grad_post_step

    fn step(self):
        """
        Perform one optimization step:
        - For each parameter with a gradient, subtract lr * grad from its buffer.
        - Optionally zero the gradient afterward.
        """

        for param_ptr in self.params:
            var ref param = param_ptr[]  # Mutably borrow the pointee
            if param.requires_grad and param.has_grad():
                grad = param.gradbox[]
                # Update the parameter values in place
                param.buffer -= grad.buffer * self.lr

            if self.zero_grad_post_step:
                param.zero_grad()

    fn zero_grad(self):
        for param_ptr in self.params:
            var ref param = param_ptr[]  # Mutably borrow the pointee
            param.zero_grad()


from common_utils import addr, addrs


# Example usage / test
fn main():
    a = Tensor.ones(2, 3, requires_grad=True)
    b = Tensor.scalar(10, requires_grad=True)

    # Wrap tensors into pointers
    params = addrs(a, b)

    # Construct optimizer
    sgd = SGD(params)
    print("\na and b\n")
    a.print()
    print()
    b.print()
    a.seed_grad(42)
    b.seed_grad(24)

    print("\npost seeding a and b grads\n")
    a.gprint()
    print()
    b.gprint()

    sgd.step()

    print("\npost stepping a and b\n")
    a.print()  # Reflects updates
    print()
    b.print()

    print("\na's and b's grad now\n")
    a.gprint()
    print()
    b.gprint()

    # Sanity check: directly viewing params inside optimizer
    sgd.params[0][].print()
    print()
    sgd.params[1][].print()
