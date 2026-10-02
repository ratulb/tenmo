from .panic import panic

struct Reduction(ImplicitlyCopyable, RegisterPassable, Writable):
    var reduction: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Reduction")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Reduction")

    def __init__(out self, reduction: Int = 0):
        self.reduction = reduction
        if reduction < 0 or reduction > 2:
            panic(
                "Reduction: must be 0=mean, 1=sum, 2=none, got "
                + String(reduction)
            )

    @implicit
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

    def __init__(out self, *, copy: Self):
        self.reduction = copy.reduction

    def is_mean(self) -> Bool:
        return self.reduction == 0

    def is_sum(self) -> Bool:
        return self.reduction == 1

    def is_none(self) -> Bool:
        return self.reduction == 2


struct WeightStrategy(
    Equatable, ImplicitlyCopyable, RegisterPassable, Writable
):
    var strategy: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Weight strategy: ", self.strategy)

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Weight strategy: ", self.strategy)

    def __init__(out self, strategy: Int = 0):
        self.strategy = strategy
        if strategy < 0 or strategy > 4:
            panic(
                "WeightStrategy: must be 0=normal, 1=uniform, 2=xavier/glorot,"
                " 3=kaiming/he, 4=zero got "
                + String(strategy)
            )

    @implicit
    def __init__(out self, strategy: String = "normal"):
        if strategy == "normal":
            self.strategy = 0
        elif strategy == "uniform":
            self.strategy = 1
        elif strategy == "xavier" or strategy == "glorot":
            self.strategy = 2
        elif strategy == "kaiming" or strategy == "he":
            self.strategy = 3
        elif strategy == "zero":
            self.strategy = 4

        else:
            self.strategy = -1
            panic(
                "WeightStrategy: must be 0=normal, 1=uniform, 2=xavier/glorot,"
                " 3=kaiming/he, 4=zero got "
                + strategy
            )

    def __init__(out self, *, copy: Self):
        self.strategy = copy.strategy

    def __eq__(self, other: Self, /) -> Bool:
        return self.strategy == other.strategy

    def __ne__(self, other: Self, /) -> Bool:
        return not self == other
