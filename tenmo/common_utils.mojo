from .shapes import Shape
from .tensor import Tensor
from .gradbox import Gradbox
from std.sys import simd_width_of
from std.sys.defines import get_defined_string
from std.logger import Level, Logger
from .net import Sequential, Linear, ReLU
from std.time import perf_counter_ns, monotonic
from std.math import cos, sin, pi
from std.os import abort
from std.utils import Variant
from std.utils.numerics import min_finite
from std.testing import assert_true
from .intarray import IntArray
from std.random import randn_float64
from .ndbuffer import NDBuffer
from std.sys import prefetch, PrefetchOptions
from std.pathlib import Path
from std.python import Python, PythonObject


comptime LOG_LEVEL = get_defined_string["LOGGING_LEVEL", "INFO"]()
comptime log = Logger[Level._from_str(LOG_LEVEL)]()

# Color codes
comptime RED: String = "\033[31m"
comptime CYAN: String = "\033[36m"
comptime MAGENTA: String = "\033[35m"
comptime BLUE: String = "\033[34m"  # Standard blue
comptime YELLOW: String = "\033[33m"  # Standard yellow
comptime RESET: String = "\033[0m"

# Bright variants (more vibrant)
comptime BRIGHT_BLUE: String = "\033[94m"


@always_inline("nodebug")
def now() -> Float64:
    return Float64(perf_counter_ns()) / 1e9


@always_inline
def binary_accuracy[
    dtype: DType,
    //,
    threshold: Scalar[dtype] = Scalar[dtype](0.5),
](pred: Tensor[dtype], target: Tensor[DType.int64]) -> Tuple[Int, Int]:
    var batch_size = pred.shape()[0]

    var prediction = pred.gt(threshold).to_dtype[DType.int64](
        requires_grad=False
    )
    var correct = prediction.eq(target).count(Scalar[DType.bool](True))
    return correct, batch_size


def multiclass_accuracy[
    dtype: DType, //
](pred: Tensor[dtype], target: Tensor[DType.int32]) -> Int:
    var correct = 0
    var batch_size = pred.shape()[0]
    var num_classes = pred.shape()[1]

    for i in range(batch_size):
        var max_idx = 0
        var max_val = pred[i, 0]
        for j in range(1, num_classes):
            if pred[i, j] > max_val:
                max_val = pred[i, j]
                max_idx = j

        if max_idx == Int(target[i]):
            correct += 1

    return correct


@always_inline("nodebug")
def log_debug(msg: String, color: String = RED):
    log.debug(color + msg + String(RESET))


@always_inline("nodebug")
def log_info(msg: String, color: String = BLUE):
    log.info(color + msg + String(RESET))


@always_inline("nodebug")
def log_warning(msg: String, color: String = YELLOW):
    log.warning(color + msg + String(RESET))


@always_inline("nodebug")
def panic(*s: String):
    var message = String(capacity=len(s))
    if len(s) > 0:
        var start = String(s[0])
        message += start.strip()
        for i in range(1, len(s)):
            var next_part = String(s[i])
            message += " " + next_part.strip()
    abort(RED + message + String(RESET))


@always_inline
def id[type: AnyType, //](t: type) -> Int:
    return Int(UnsafePointer(to=t).as_immutable())


@always_inline
def addr[
    mut: Bool, origin: Origin[mut=mut], type: AnyType, //
](t: type) -> UnsafePointer[type, origin]:
    return UnsafePointer(to=t).mut_cast[mut]().unsafe_origin_cast[origin]()


@always_inline
def addrs[
    mut: Bool, origin: Origin[mut=mut], type: AnyType, //
](*ts: type) -> List[UnsafePointer[type, origin]]:
    l = List[UnsafePointer[type, origin]](capacity=len(ts))
    for t in ts:
        l.append(
            UnsafePointer(to=t).mut_cast[mut]().unsafe_origin_cast[origin]()
        )
    return l^


def copy[
    is_mut: Bool, origin: Origin[mut=is_mut], dtype: DType, //
](
    src: UnsafePointer[Scalar[dtype], origin],
    dest: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
):
    """
    General-purpose optimized copy with smart defaults.
    """

    print("Utils copy")
    if count <= 0:
        return

    comptime simd_width = simd_width_of[dtype]()

    # For small copies, skip prefetching overhead
    if count < 1024:
        var i = 0
        var vec_end = (count // simd_width) * simd_width

        while i < vec_end:
            dest.store[width=simd_width](i, src.load[width=simd_width](i))
            i += simd_width

        while i < count:
            dest[i] = src[i]
            i += 1

        return

    # For large copies, use prefetching
    comptime unrolled = 8
    comptime chunk_size = simd_width * unrolled
    comptime prefetch_ahead = chunk_size * 4
    comptime prefetch_opts = PrefetchOptions().for_read().low_locality().to_data_cache()

    var i = 0
    var end = (count // chunk_size) * chunk_size

    while i < end:
        if i + prefetch_ahead < count:
            prefetch[prefetch_opts](src + i + prefetch_ahead)

        comptime for j in range(unrolled):
            var offset = i + j * simd_width
            dest.store[width=simd_width](
                offset, src.load[width=simd_width](offset)
            )

        i += chunk_size

    # Vectorized remainder
    var vec_end = (count // simd_width) * simd_width
    while i < vec_end:
        dest.store[width=simd_width](i, src.load[width=simd_width](i))
        i += simd_width

    # Scalar tail
    while i < count:
        dest[i] = src[i]
        i += 1


def is_null[
    type: AnyType, //
](ptr: UnsafePointer[type, ImmutAnyOrigin]) -> Bool:
    return ptr.__bool__() == False


struct IDGen(RegisterPassable):
    @always_inline
    @staticmethod
    def generate_id() -> UInt:
        # Use both perf_counter and monotonic for additional entropy
        perf_time = perf_counter_ns()
        mono_time = monotonic()

        # Combine them in a way that preserves uniqueness
        # Use XOR to mix the values
        return perf_time ^ (mono_time << 32)


struct Epsilon[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype == DType.float32:
            return Scalar[Self.dtype](1e-7)
        elif Self.dtype == DType.float64:
            return Scalar[Self.dtype](1e-12)
        elif Self.dtype == DType.float16:
            return rebind[Scalar[Self.dtype]](min_finite[DType.float16]())
        elif Self.dtype == DType.int32:
            return rebind[Scalar[Self.dtype]](min_finite[DType.int32]())
        elif Self.dtype == DType.int64:
            return rebind[Scalar[Self.dtype]](min_finite[DType.int64]())
        elif Self.dtype == DType.int8:
            return rebind[Scalar[Self.dtype]](min_finite[DType.int8]())
        elif Self.dtype == DType.uint8:
            return rebind[Scalar[Self.dtype]](min_finite[DType.uint8]())
        elif Self.dtype == DType.int16:
            return rebind[Scalar[Self.dtype]](min_finite[DType.int16]())
        elif Self.dtype == DType.bool:
            return Scalar[Self.dtype](False)  #!
        else:
            panic("Epsilon value not supported for: ", String(Self.dtype))
            return Scalar[Self.dtype](0)


struct One[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            return Scalar[Self.dtype](1.0)
        else:
            return Scalar[Self.dtype](1)


struct Zero[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            return Scalar[Self.dtype](0.0)
        else:
            return Scalar[Self.dtype](0)


@always_inline("nodebug")
def inf[dtype: DType]() -> Scalar[dtype]:
    """Gets a +inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The +inf value of the given dtype.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Only floating point dtypes support +inf."

    comptime if dtype == DType.bfloat16:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<bf16>`,
        )
    elif dtype == DType.float16:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f16>`,
        )
    elif dtype == DType.float32:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f32>`,
        )
    elif dtype == DType.float64:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f64>`,
        )
    else:
        comptime assert False, "unsupported float type"


@always_inline("nodebug")
def isinf[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return inf[dtype]() == value


@always_inline("nodebug")
def isnan[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return nan[dtype]() == value


@always_inline("nodebug")
def nan[dtype: DType]() -> Scalar[dtype]:
    """Gets a NaN value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The NaN value of the given dtype.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Only floating point dtypes support NaN."

    comptime if dtype == DType.float32:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"nan"> : !pop.scalar<f32>`,
        )
    elif dtype == DType.float64:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"nan"> : !pop.scalar<f64>`,
        )
    else:
        comptime assert False, "unsupported float type"


# Helper
def do_assert[
    dtype: DType, //
](a: Tensor[dtype], b: Tensor[dtype], msg: String) raises:
    shape_mismatch = String("{0}: shape mismatch {1} vs {2}")
    tensors_not_equal = String("{}: values mismatch")
    assert_true(
        a.shape() == b.shape(), shape_mismatch.format(msg, a.shape(), b.shape())
    )
    assert_true((a == b), tensors_not_equal.format(msg))


# Helper
def assert_grad[
    dtype: DType, //
](t: Tensor[dtype], expected: Tensor[dtype], label: String) raises:
    assert_true(
        (t.grad() == expected),
        String("grad assertion failed for " + label),
    )


@fieldwise_init
struct NewAxis(Copyable & Movable):  # Empty struct as a sentinel
    pass


comptime Idx = Variant[Int, IntArray, Slice, NewAxis]

comptime newaxis = Idx(NewAxis())


@always_inline("nodebug")
def i(value: Int) -> Idx:
    return Idx(value)


@always_inline("nodebug")
def il(index_list: IntArray) -> Idx:
    return Idx(index_list)


@always_inline("nodebug")
def il(*indices: Int) -> Idx:
    intarray = IntArray.with_capacity(len(indices))
    for i in range(len(indices)):
        intarray.append(indices[i])
    return Idx(intarray)


@always_inline("nodebug")
def s() -> Idx:
    return s(None, None, None)


@always_inline("nodebug")
def s(end: Int) -> Idx:
    return Idx(slice(end))


@always_inline("nodebug")
def s(start: Int, end: Int) -> Idx:
    return Idx(slice(start, end))


@always_inline("nodebug")
def s(start: Optional[Int], end: Optional[Int], step: Optional[Int]) -> Idx:
    return Idx(slice(start, end, step))


struct Slicer:
    @staticmethod
    @always_inline("nodebug")
    def slice(
        slice: Slice, end: Int, start: Int = 0, step: Int = 1
    ) -> Tuple[Int, Int, Int]:
        _start, _end, _step = (
            slice.start.or_else(start),
            slice.end.or_else(end),
            slice.step.or_else(step),
        )
        return _start, _end, _step


# Utility repeat function
def str_repeat(s: String, n: Int) -> String:
    if n <= 0:
        return ""
    var parts = List[String]()
    for _ in range(n):
        parts.append(s)
    return StringSlice("").join(parts)


def print_summary[
    dtype: DType
](
    mod: Sequential[dtype], sample_input: Optional[Tensor[dtype]] = None
) where dtype.is_floating_point():
    # Table headers
    var headers = List[String]()
    headers.append("Name")
    headers.append("Type")
    headers.append("Input Shape")
    headers.append("Output Shape")
    headers.append("Params")
    headers.append("Trainable")

    var rows = List[List[String]]()
    rows.append(headers.copy())

    var total_params = 0
    var trainable_params = 0

    # If sample_input is provided → run a dry forward pass to get shapes
    var x = sample_input
    var current_shape = "(?, ?)"

    for i in range(len(mod.modules)):
        m = mod.modules[i].copy()
        var name = "Layer" + String(i)

        if m.layer.isa[Linear[dtype]]():
            var l = m.layer[Linear[dtype]].copy()

            # Infer input/output shapes
            var in_features = l.weight.shape()[0]
            var out_features = l.weight.shape()[1]

            var input_shape = "(?, " + String(in_features) + ")"
            var output_shape = "(?, " + String(out_features) + ")"

            if x:
                input_shape = String(x.value().shape())
                x = Optional(m(x.value()))
                output_shape = String(x.value().shape())

            current_shape = output_shape

            # Params
            var params = (
                l.weight.shape().num_elements() + l.bias.shape().num_elements()
            )
            total_params += params
            if l.weight.requires_grad or l.bias.requires_grad:
                trainable_params += params

            rows.append(
                [
                    name,
                    "Linear",
                    input_shape,
                    output_shape,
                    String(params),
                    String(l.weight.requires_grad or l.bias.requires_grad),
                ]
            )

        elif m.layer.isa[ReLU[dtype]]():
            var input_shape = current_shape
            var output_shape = current_shape

            if x:
                input_shape = String(x.value().shape())
                x = Optional(m(x.value()))
                output_shape = String(x.value().shape())
                current_shape = output_shape

            rows.append([name, "ReLU", input_shape, output_shape, "0", "False"])

    # Compute column widths
    var widths = List[Int]()
    for j in range(len(headers)):
        var maxw = 0
        for row in rows:
            if len(row[j]) > maxw:
                maxw = len(row[j])
        widths.append(maxw)

    # Print horizontal rule
    def print_rule(read widths: List[Int]):
        var line = ""
        for w in widths:
            line += "+" + str_repeat("-", w + 2)
        line += "+"
        print(line)

    # Print table
    print_rule(widths)
    for idx in range(len(rows)):
        var line = ""
        for j in range(len(rows[idx])):
            var val = rows[idx][j]
            line += "| " + val + str_repeat(" ", widths[j] - len(val)) + " "
        line += "|"
        print(line)
        print_rule(widths)

    # Footer summary
    var non_trainable_params = total_params - trainable_params
    print("\nSummary:")
    print("  Total params:        ", total_params)
    print("  Trainable params:    ", trainable_params)
    print("  Non-trainable params:", non_trainable_params)


def print_buffer[
    dtype: DType
](
    read buffer: NDBuffer[dtype],
    mut indices: List[Int],
    level: Int,
    num_first: Int = 10,
    num_last: Int = 10,
):
    if buffer.buffer.size == 0 and buffer.device_state == None:
        # if buffer.numels() == 0:
        print("  Empty")
        return
    if buffer.rank() == 0:  # Tensor with Shape ()
        print(buffer[[]])
        return
    current_dim = len(indices)
    indent = " " * (level * 2)

    if current_dim >= buffer.rank():
        print(
            "ERROR: current_dim (",
            current_dim,
            ") >= ndim (",
            buffer.rank(),
            ")",
        )
        return

    size = buffer.shape[current_dim]

    if size < 0 or size > 1_000_000:
        print(
            "ERROR: suspicious size: ",
            size,
            "at dim ",
            current_dim,
            String(buffer.shape),
        )
        return

    # Base case: last dimension (print actual elements)
    if current_dim == buffer.rank() - 1:
        print(indent + "[", end="")

        for i in range(size):
            if i < num_first:
                indices.append(i)
                print(buffer[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")
            elif i == num_first and size > num_first + num_last:
                print("..., ", end="")
            elif i >= size - num_last:
                indices.append(i)
                print(buffer[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")

        print("]", end="")

    else:
        print(indent + "[")
        for i in range(size):
            if i < num_first:
                indices.append(i)
                print_buffer(buffer, indices, level + 1, num_first, num_last)
                _ = indices.pop()
            elif i == num_first and size > num_first + num_last:
                print(indent + "  ...,")
            elif i >= size - num_last:
                indices.append(i)
                print_buffer(buffer, indices, level + 1, num_first, num_last)
                _ = indices.pop()

            # Print comma and newline for all but last element
            if i != size - 1 and (i < num_first or i >= size - num_last):
                print(",")
            # Special case: last element needs newline before closing bracket
            elif i == size - 1:
                print()  # Newline before closing bracket

        print(indent + "]", end="")


def download(url: String, save_to: String) raises:
    try:
        var urllib = Python.import_module("urllib.request")
        var response = urllib.urlopen(url)
        var content: String = String(response.read().decode("utf-8"))
        var file = Path(save_to)
        file.write_text(content)
    except e:
        print(e)
        raise e^


def pystr(s: String) raises -> PythonObject:
    return Python.str(s)

comptime DEFAULT_SPLITTER = r'([,.:;?_!"()\']|--|\s)'
comptime DEFAULT_SUBSTITUTION: Tuple[StaticString, StaticString] = (
    r'\s+([,.:;?!"()\'])',
    r"\1",
)
comptime DEFAULT_UNK = "<|unk|>"
comptime END_OF_TEXT = "<|endoftext|>"

@fieldwise_init
struct SimpleTokenizer[
    splitter: StaticString = DEFAULT_SPLITTER,
    substitution: Tuple[StaticString, StaticString] = DEFAULT_SUBSTITUTION,
    UNK: StaticString = DEFAULT_UNK,
    end_of_text: StaticString = END_OF_TEXT,
](Sized & ImplicitlyCopyable & Movable):
    var str_to_int: Dict[String, Int]
    var int_to_str: Dict[Int, String]
    var regex_parser: PythonObject

    def __init__(out self, var vocab: Dict[String, Int]) raises:
        self.int_to_str = {item.value: item.key for item in vocab.items()}
        self.str_to_int = vocab^
        self.regex_parser = Python.import_module("re")

    def __copyinit__(out self, copy: Self):
        self.int_to_str = copy.int_to_str.copy()
        self.str_to_int = copy.str_to_int.copy()
        self.regex_parser = copy.regex_parser.copy()

    @staticmethod
    def from_text_lines[
        mut: Bool, //, origin: Origin[mut=mut]
    ](lines: List[StringSlice[origin=origin]]) raises -> Self:
        try:
            var all_lines = " ".join(lines)
            var re = Python.import_module("re")
            var splitted = re.split(Self.splitter, all_lines)
            splitted = Python.list([item for item in splitted if item.strip()])
            var py = Python.import_module("builtins")
            var unique_words = py.sorted(py.set(splitted))
            var extension: PythonObject = [Self.end_of_text, Self.UNK]
            unique_words.extend(extension)
            var vocab = {
                String(token): Int(index)
                for index, token in enumerate(unique_words.__iter__())
            }
            return Self(vocab^)
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_text_lines_lower_case[
        mut: Bool, //, origin: Origin[mut=mut]
    ](lines: List[StringSlice[origin=origin]]) raises -> Self:
        try:
            var all_lines = " ".join(lines)
            var re = Python.import_module("re")

            # 1) Lowercase everything before splitting
            var py_str = Python.import_module("builtins").str(all_lines).lower()

            # 2) Keep only word characters (letters, digits, underscore)
            #    Split on anything that is NOT a word character or apostrophe
            var splitted = re.split(r"[^a-z0-9']+", py_str)
            splitted = Python.list([item for item in splitted if item.strip()])

            var py = Python.import_module("builtins")
            var unique_words = py.sorted(py.set(splitted))
            var extension: PythonObject = [Self.end_of_text, Self.UNK]
            unique_words.extend(extension)
            var vocab = {
                String(token): Int(index)
                for index, token in enumerate(unique_words.__iter__())
            }
            return Self(vocab^)
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_text_lines_strip_html[
        mut: Bool, //, origin: Origin[mut=mut]
    ](lines: List[StringSlice[origin=origin]]) raises -> Self:
        try:
            var all_lines = " ".join(lines)
            var re = Python.import_module("re")
            var py = Python.import_module("builtins")
            var py_str = py.str(all_lines)

            # 1) Lowercase
            py_str = py_str.lower()

            # 2) Strip HTML tags e.g. <br />, <a href=...>
            py_str = re.sub(r'<[^>]+>', ' ', py_str)

            # 3) Strip URLs
            py_str = re.sub(r'http\S+|www\.\S+', ' ', py_str)

            # 4) Strip digits — numbers add huge noise to vocab
            py_str = re.sub(r'\d+', ' ', py_str)

            # 5) Keep only plain letters and apostrophes
            py_str = re.sub(r"[^a-z']+", ' ', py_str)

            # 6) Strip floating apostrophes (not contractions)
            py_str = re.sub(r"(?<!\w)'|'(?!\w)", ' ', py_str)

            # 7) Collapse whitespace
            #py_str = re.sub(r'\s+', ' ', py_str)

            #var splitted = py_str.split(' ')
            #splitted = py.list([item for item in splitted if item.strip()])

            #var unique_words = py.sorted(py.set(splitted))"""
            # 7) Collapse whitespace and split — no empty tokens
            py_str = re.sub(r'\s+', ' ', py_str).strip()
            var splitted = py_str.split(' ')

            var unique_words = py.sorted(py.set(splitted))

            var extension: PythonObject = [Self.end_of_text, Self.UNK]
            unique_words.extend(extension)
            var vocab = {
                String(token): Int(index)
                for index, token in enumerate(unique_words.__iter__())
            }
            return Self(vocab^)
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_text_lines_min_freq[
        mut: Bool, //, origin: Origin[mut=mut]
    ](lines: List[StringSlice[origin=origin]]) raises -> Self:
        try:
            var all_lines = " ".join(lines)
            var re = Python.import_module("re")
            var py = Python.import_module("builtins")
            var collections = Python.import_module("collections")
            var py_str = py.str(all_lines)

            # Clean
            py_str = py_str.lower()
            py_str = re.sub(r'<[^>]+>', ' ', py_str)
            py_str = re.sub(r'http\S+|www\.\S+', ' ', py_str)
            py_str = re.sub(r'\d+', ' ', py_str)
            py_str = re.sub(r"[^a-z']+", ' ', py_str)
            py_str = re.sub(r"(?<!\w)'|'(?!\w)", ' ', py_str)
            py_str = re.sub(r'\s+', ' ', py_str).strip()

            var splitted = py_str.split(' ')
            var freq = collections.Counter(splitted)
            var min_freq: PythonObject = 5

            # most_common returns all items sorted by count — filter via Python eval
            var filter_fn = Python.evaluate("lambda f, mf: [w for w, c in f.items() if c >= mf]")
            var filtered = filter_fn(freq, min_freq)
            var unique_words = py.sorted(filtered)
            var extension: PythonObject = [Self.end_of_text, Self.UNK]
            unique_words.extend(extension)



            # Count frequencies and keep only words appearing >= min_freq times
            _="""var freq = collections.Counter(splitted)
            var min_freq: Int = 2
            var filtered = py.list(
                [w for w in freq if freq[w] >= min_freq]
            )

            var unique_words = py.sorted(filtered)
            var extension: PythonObject = [Self.end_of_text, Self.UNK]
            unique_words.extend(extension)"""
            var vocab = {
                String(token): Int(index)
                for index, token in enumerate(unique_words.__iter__())
            }
            return Self(vocab^)
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_file(file_path: String) raises -> Self:
        try:
            var path = Path(file_path)
            var content = path.read_text()
            return Self.from_text_lines(content.splitlines())
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_url(url: String) raises -> Self:
        try:
            var urllib = Python.import_module("urllib.request")
            var response = urllib.urlopen(url)
            var content = String(response^.read().decode("utf-8"))
            return Self.from_text_lines(content.splitlines())
        except e:
            print(e)
            raise e^

    def encode(self, text: String) raises -> List[Int]:
        var splitted = self.regex_parser.split(Self.splitter, text)
        splitted = Python.list([item for item in splitted if item.strip()])
        var token_ids = List[Int](capacity=len(splitted))
        for token in splitted:
            var token_str = String(token)
            token_ids.append(
                self.str_to_int[token_str] if token_str
                in self.str_to_int else self.str_to_int[Self.UNK]
            )
        return token_ids^

    def encode_lower_case(self, text: String) raises -> List[Int]:
        var re = self.regex_parser
        # Lowercase input before encoding
        var py_text = Python.import_module("builtins").str(text).lower()
        var splitted = re.split(r"[^a-z0-9']+", py_text)
        splitted = Python.list([item for item in splitted if item.strip()])
        var token_ids = List[Int](capacity=len(splitted))
        for token in splitted:
            var token_str = String(token)
            token_ids.append(
                self.str_to_int[token_str] if token_str
                in self.str_to_int else self.str_to_int[Self.UNK]
            )
        return token_ids^

    def encode_strip_html(self, text: String) raises -> List[Int]:
        ref re = self.regex_parser
        var py = Python.import_module("builtins")
        var py_str = py.str(text)

        py_str = py_str.lower()
        py_str = re.sub(r'<[^>]+>', ' ', py_str)
        py_str = re.sub(r'http\S+|www\.\S+', ' ', py_str)
        py_str = re.sub(r'\d+', ' ', py_str)
        py_str = re.sub(r"[^a-z']+", ' ', py_str)
        py_str = re.sub(r"(?<!\w)'|'(?!\w)", ' ', py_str)
        #py_str = re.sub(r'\s+', ' ', py_str)

        #var splitted = py_str.split(' ')
        #splitted = Python.list([item for item in splitted if item.strip()])
        py_str = re.sub(r'\s+', ' ', py_str).strip()
        var splitted = py_str.split(' ')

        var token_ids = List[Int](capacity=len(splitted))
        for token in splitted:
            var token_str = String(token)
            token_ids.append(
                self.str_to_int[token_str] if token_str
                in self.str_to_int else self.str_to_int[Self.UNK]
            )
        return token_ids^

    def decode(self, token_ids: List[Int]) raises -> String:
        var text = " ".join([self.int_to_str[id] for id in token_ids])
        text = String(
            self.regex_parser.sub(
                Self.substitution[0], Self.substitution[1], text^
            )
        )
        return text^

    def __len__(self) -> Int:
        return len(self.int_to_str)
