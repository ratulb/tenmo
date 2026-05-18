from tenmo.common_utils import panic


struct Reduction(ImplicitlyCopyable, RegisterPassable):
    var reduction: Int

    def __init__(out self, reduction: Int = 0):
        self.reduction = reduction
        if reduction < 0 or reduction > 2:
            panic(
                "Reduction: must be 0=mean, 1=sum, 2=none, got "
                + String(reduction)
            )

    def __init__(out self, reduction: String):
        if reduction == "mean":
            self.reduction = 0
        elif reduction == "sum":
            self.reduction = 1
        elif reduction == "none":
            self.reduction = 2
        else:
            self.reduction = -1
            panic(
                "Reduction: must be 'mean', 'sum', or 'none', got '"
                + reduction
                + "'"
            )

    def __copyinit__(out self, copy: Self):
        self.reduction = copy.reduction

    def is_mean(self) -> Bool:
        return self.reduction == 0

    def is_sum(self) -> Bool:
        return self.reduction == 1

    def is_none(self) -> Bool:
        return self.reduction == 2
