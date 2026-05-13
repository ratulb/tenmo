from std.pathlib import Path
from std.python import Python, PythonObject
from tenmo.nlp import IMDBTextCleaner

comptime DEFAULT_SPLITTER = r'([,.:;?_!"()\']|--|\s)'
comptime DEFAULT_SUBSTITUTION: Tuple[StaticString, StaticString] = (
    r'\s+([,.:;?!"()\'])',
    r"\1",
)
comptime DEFAULT_UNK = "<|unk|>"
comptime END_OF_TEXT = "<|endoftext|>"

comptime DefaultTokenizer = SimpleTokenizer[
    DEFAULT_SPLITTER, DEFAULT_SUBSTITUTION, DEFAULT_UNK, END_OF_TEXT
]


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
    var text_cleaner: IMDBTextCleaner
    var max_n: Int

    def __init__(
        out self,
        var vocab: Dict[String, Int],
        text_cleaner: IMDBTextCleaner,
        max_n: Int = 1,
    ) raises:
        self.int_to_str = {item.value: item.key for item in vocab.items()}
        self.str_to_int = vocab^
        self.text_cleaner = text_cleaner
        self.max_n = max_n
        self.regex_parser = Python.import_module("re")

    def __copyinit__(out self, copy: Self):
        self.int_to_str = copy.int_to_str.copy()
        self.str_to_int = copy.str_to_int.copy()
        self.text_cleaner = copy.text_cleaner
        self.max_n = copy.max_n
        self.regex_parser = copy.regex_parser.copy()

    def __moveinit__(out self, deinit take: Self):
        self.int_to_str = take.int_to_str^
        self.str_to_int = take.str_to_int^
        self.text_cleaner = take.text_cleaner
        self.max_n = take.max_n
        self.regex_parser = take.regex_parser^

    @staticmethod
    def ngramify(
        words:   PythonObject,
        max_n:   Int,
        cleaner: IMDBTextCleaner,
    ) raises -> PythonObject:
        var py     = Python.import_module("builtins")
        var n      = len(words)
        var result = py.list(words)   # unigrams — full word list, no stopword filter

        if max_n >= 2:
            # n-grams use content words only — stopwords filtered here
            var cw      = cleaner.content_words(words)
            var nc      = len(cw)
            var bigrams = py.list()
            for k in range(nc - 1):
                bigrams.append(cw[k] + py.str('_') + cw[k + 1])
            result.extend(bigrams)

        if max_n >= 3:
            var cw       = cleaner.content_words(words)
            var nc       = len(cw)
            var trigrams = py.list()
            for k in range(nc - 2):
                trigrams.append(
                    cw[k]     + py.str('_')
                    + cw[k+1] + py.str('_')
                    + cw[k+2]
                )
            result.extend(trigrams)

        return result

    @staticmethod
    def from_text_lines(
        lines: List[String],
        text_cleaner: IMDBTextCleaner,
        min_freq: Int = 5,
        max_n: Int = 1,
    ) raises -> Self:
        """Build vocabulary from text lines using any TextCleaner.

        Orchestrates: clean_text → ngramify → frequency filter → sentinels → vocab.

        Args:
            lines:    Raw text lines — one per document.
            text_cleaner:  Any TextCleaner implementor.
            min_freq: Minimum token frequency to include in vocab.
            max_n:    Maximum n-gram size.

        Returns:
            SimpleTokenizer parameterized on the same CleanerType as `cleaner`.
        """
        var py = Python.import_module("builtins")
        var collections = Python.import_module("collections")
        var freq = collections.Counter()

        for line in lines:
            var words = text_cleaner.clean_text(line)
            var tokens = Self.ngramify(words, max_n, text_cleaner)
            freq.update(tokens)

        var filter_fn = Python.evaluate(
            "lambda f, mf: [w for w, c in f.items() if c >= mf]"
        )
        var min_freq_py: PythonObject = min_freq
        var filtered = filter_fn(freq, min_freq_py)
        var unique_words = py.sorted(filtered)

        var extension: PythonObject = [Self.end_of_text, Self.UNK]
        unique_words.extend(extension)

        var vocab = {
            String(token): Int(index)
            for index, token in enumerate(unique_words.__iter__())
        }
        return Self(vocab^, text_cleaner, max_n)

    @staticmethod
    def from_file(
        file_path: String,
        text_cleaner: IMDBTextCleaner,
        min_freq: Int = 5,
        max_n: Int = 1,
    ) raises -> Self:
        try:
            var path = Path(file_path)
            var content = path.read_text()
            var lines = List[String]()
            for line in content.splitlines():
                lines.append(String(line))
            return Self.from_text_lines(
                lines^, text_cleaner, min_freq, max_n
            )
        except e:
            print(e)
            raise e^

    @staticmethod
    def from_url(
        url: String,
        text_cleaner: IMDBTextCleaner,
        min_freq: Int = 5,
        max_n: Int = 1,
    ) raises -> Self:
        try:
            var urllib = Python.import_module("urllib.request")
            var response = urllib.urlopen(url)
            var content = String(response^.read().decode("utf-8"))
            var lines = List[String]()
            for line in content.splitlines():
                lines.append(String(line))
            return Self.from_text_lines(
                lines^, text_cleaner, min_freq, max_n
            )
        except e:
            print(e)
            raise e^

    def encode(self, text: String) raises -> List[Int]:
        var words = self.text_cleaner.clean_text(text)
        var tokens = Self.ngramify(words, self.max_n, self.text_cleaner)
        var token_ids = List[Int](capacity=len(tokens))
        for token in tokens:
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
