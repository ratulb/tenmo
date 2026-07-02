"""Character-level tokenizer for nanoGPT.

Builds a vocabulary from all unique Unicode codepoints in a training corpus,
then maps each codepoint to/from a dense integer ID.
"""

from std.collections import Set
from std.pathlib import Path
from std.python import Python


@fieldwise_init
struct SimpleTokenizer(ImplicitlyCopyable & Movable & Sized & Writable):
    """Character-level tokenizer mapping codepoints to ``Int`` token IDs.

    Supports ``len()`` via the ``Sized`` trait (returns vocabulary size).

    Inspired by Andrej Karpathy's nanoGPT tokenizer. Collects every unique
    Unicode codepoint from a training text, assigns each a dense index in
    sorted order, and provides encode/decode methods.

    Encode returns ``List[Int]`` (signed 64-bit). Convert to
    ``Tensor[DType.uint32]`` for compact storage in model training.

    Examples:
        var text = String("hello")
        var tok = SimpleTokenizer(text)
        print(tok.encode("hello"))   # [1, 0, 2, 2, 3]
        print(tok.decode([1, 0, 2, 2, 3]))  # "hello"
    """

    var stoi: Dict[String, Int]
    """Mapping from single-character strings (codepoints) to token IDs."""

    var itos: Dict[Int, String]
    """Mapping from token IDs back to single-character strings."""

    def __init__(out self, text: String):
        """Build vocabulary from all unique codepoints in *text*.

        Codepoints are collected, deduplicated, sorted, and assigned dense
        ``Int`` IDs in ascending order.

        Args:
            text: Training corpus to scan for unique Unicode codepoints.
        """
        var codepoints = [
            code_point
            for code_point in Set(
                [codepoint.to_u32() for codepoint in text.codepoints()]
            )
        ]
        sort(codepoints)
        var stoi = {
            chr(Int(codepoint)): i for i, codepoint in enumerate(codepoints^)
        }
        var itos = {item.value: item.key for item in stoi.items()}
        self.stoi = stoi^
        self.itos = itos^

    @staticmethod
    def from_file(
        file_path: String,
    ) raises -> Self:
        """Read a local text file and build a tokenizer for its codepoints.

        The file is read as UTF-8 via ``Path.read_text``. The full content
        becomes the training corpus for vocabulary construction, exactly as
        with the string constructor.

        Args:
            file_path: Path to a UTF-8 encoded text file on disk.

        Returns:
            A ``SimpleTokenizer`` whose vocabulary covers every codepoint
            present in the file.

        Raises:
            Error if the file does not exist, cannot be read, or is not
            valid UTF-8.
        """
        try:
            var path = Path(file_path)
            var content = path.read_text()
            return Self(content^)
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_url(
        url: String,
    ) raises -> Self:
        """Download text from *url* and build a tokenizer for its codepoints.

        Uses Python's ``urllib.request`` under the hood. The downloaded text
        is used as the training corpus to collect the vocabulary, exactly as
        with the string constructor.

        Args:
            url: HTTP/HTTPS URL pointing to a UTF-8 text file
                (e.g. a raw GitHub text or a public corpus).

        Returns:
            A ``SimpleTokenizer`` whose vocabulary covers every codepoint
            present in the downloaded text.

        Raises:
            Error if the URL is unreachable, the HTTP request fails, or the
            response cannot be decoded as UTF-8.
        """
        try:
            var urllib = Python.import_module("urllib.request")
            var response = urllib.urlopen(url)
            var content = String(response^.read().decode("utf-8"))
            return Self(content^)
        except e:
            print(e)
            raise e^

    def __init__(out self, *, copy: Self):
        """Copy constructor — deep-copies both dicts."""
        self.stoi = copy.stoi.copy()
        self.itos = copy.itos.copy()

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, s: StringSlice[origin]) raises -> List[Int]:
        """Encode a ``StringSlice`` into a list of token IDs.

        Each Unicode codepoint in *s* is looked up in the vocabulary and
        replaced by its ``Int`` ID. Raises if a codepoint was not seen
        during training.

        Args:
            s: The ``StringSlice`` (or ``String``) to tokenize.

        Returns:
            List of ``Int`` token IDs, one per codepoint.

        Raises:
            Error if any codepoint in *s* is absent from the vocabulary.
        """
        return [
            self.stoi[chr(Int(codepoint.to_u32()))]
            for codepoint in s.codepoints()
        ]

    def decode(self, l: List[Int]) raises -> String:
        """Decode a list of token IDs back into a string.

        Each ``Int`` ID is mapped back to its single-character string and
        all pieces are concatenated.

        Args:
            l: Sequence of ``Int`` token IDs produced by ``encode``.

        Returns:
            The reconstructed string.
        """
        return StringSlice("").join([self.itos[i] for i in l])

    def __len__(self) -> Int:
        """Return vocabulary size (number of unique codepoints)."""
        return len(self.itos)

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        var keys = [key for key in self.stoi.keys()]
        sort(keys)
        writer.write(StringSlice("").join(keys^))

    def encode_as[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
        dtype: DType,
    ](self, s: StringSlice[origin]) raises -> List[Scalar[dtype]]:
        return [
            Scalar[dtype](self.stoi[chr(Int(codepoint.to_u32()))])
            for codepoint in s.codepoints()
        ]

    def decode_from[
        mut: Bool,
        dtype: DType,
        //,
        origin: Origin[mut=mut],
    ](self, l: Span[Scalar[dtype], origin]) raises -> String:
        return StringSlice("").join([self.itos[Int(i)] for i in l])
