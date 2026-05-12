from std.python import Python, PythonObject


@fieldwise_init
struct IMDBTextCleaner(RegisterPassable & ImplicitlyCopyable):
    fn clean_text(self, text: String) raises -> PythonObject:
        var re = Python.import_module("re")
        var py = Python.import_module("builtins")
        var py_str = py.str(text)

        # 1. Lowercase — Good/GOOD/good all become good
        py_str = py_str.lower()

        # 2. Strip HTML tags — <br />, <p>, <a href="...">, </div> etc.
        py_str = re.sub(r"<[^>]+>", " ", py_str)

        # 3. Strip URLs — http://example.com or www.example.com
        py_str = re.sub(r"http\S+|www\.\S+", " ", py_str)

        # 4. Strip digit sequences — 2024, 10, 3rd act, etc.
        py_str = re.sub(r"\d+", " ", py_str)

        # 5. Keep only lowercase letters and apostrophes — everything else → space
        py_str = re.sub(r"[^a-z']+", " ", py_str)

        # 6. Strip stray apostrophes not part of a contraction.
        py_str = re.sub(r"(?<!\w)'|'(?!\w)", " ", py_str)

        # 7. Collapse multiple spaces/newlines → single space, strip ends
        py_str = re.sub(r"\s+", " ", py_str).strip()

        # Split on single space — no empty strings after strip+collapse
        return py_str.split(" ")
