from std.python import Python, PythonObject


# =============================================================================
# IMDBTextCleaner
# =============================================================================

# Stopwords filtered BEFORE n-gram assembly only.
# These high-frequency low-signal words create noisy bigrams/trigrams
# (e.g. "on_it_it", "is_the", "a_film") that pollute the vocabulary
# without adding semantic value.
# Unigrams are NOT filtered — "the", "a" etc. remain in unigram vocab
# since their presence/absence carries some signal for bag-of-words models.
comptime STOPWORDS = """
{
    'the','a','an','and','or','but','in','on','at','to',
    'for','of','with','by','from','is','it','as','be',
    'was','are','were','has','had','have','this','that',
    'i','he','she','they','we','you','his','her','their',
    'its','my','our','your','so','do','did','not','no',
    'if','up','out','about','than','into','then','there',
    'what','which','who','how','when','where','will','would',
    'could','should','may','might','also','just','been','after',
    'before','more','all','one','can','get','got','him','them'
}
"""


@fieldwise_init
struct IMDBTextCleaner(RegisterPassable & ImplicitlyCopyable):
    """Text cleaner for IMDB movie review data.

    Designed for web-scraped English review text. Strips HTML, URLs,
    digits and non-alphabetic characters. Preserves apostrophes for
    English contractions (don't, it's, they're).

    Two-phase design:
        clean_text(text)        — cleans one document, returns Python word list.
                                  Single source of truth for the cleaning pipeline.
                                  Used by SimpleTokenizer.from_text_lines,
                                  SimpleTokenizer.encode, and build_datasets.

        content_words(words)    — filters stopwords from a cleaned word list
                                  for use in n-gram assembly only.
                                  Unigrams always use the full cleaned word list.

    Cleaning pipeline (applied in order in clean_text):
        1. Lowercase              — collapse case variants (Good/GOOD → good)
        2. Strip HTML tags        — remove <br />, <a href=...> etc.
        3. Strip URLs             — remove http://... and www....
        4. Strip digit sequences  — remove years, scores, phone numbers
        5. Keep only [a-z']       — strip punctuation, symbols, non-ASCII
        6. Strip stray apostrophes — remove apostrophes not in contractions
        7. Collapse whitespace    — normalise spaces/newlines
        8. Minimum word length    — drop words shorter than 2 characters
                                    eliminates single-char noise tokens that
                                    create meaningless n-grams like 'a_i_s'
    """

    fn clean_text(self, text: String) raises -> PythonObject:
        """Clean ONE document and return a Python word list.

        This is the single source of truth for what 'clean' means.
        SimpleTokenizer.ngramify, from_text_lines, encode and
        build_datasets all call this method — cleaning is never
        duplicated elsewhere.

        Args:
            text: Raw review string.

        Returns:
            Python list of clean lowercase word strings, each at
            least 2 characters long. No empty strings.
        """
        var re     = Python.import_module("re")
        var py     = Python.import_module("builtins")
        var py_str = py.str(text)

        # 1. Lowercase — Good/GOOD/good all become good
        py_str = py_str.lower()

        # 2. Strip HTML tags — <br />, <p>, <a href="...">, </div> etc.
        #    [^>]+ matches one or more characters that are not >
        py_str = re.sub(r"<[^>]+>", " ", py_str)

        # 3. Strip URLs — http://example.com or www.example.com
        #    \S+ matches any non-whitespace run after http or www.
        py_str = re.sub(r"http\S+|www\.\S+", " ", py_str)

        # 4. Strip digit sequences — 2024, 10, 3rd etc.
        #    \d+ matches one or more consecutive digit characters
        py_str = re.sub(r"\d+", " ", py_str)

        # 5. Keep only lowercase letters and apostrophes
        #    [^a-z']+ matches anything that is NOT a-z or apostrophe
        py_str = re.sub(r"[^a-z']+", " ", py_str)

        # 6. Strip stray apostrophes not part of a contraction.
        #    Preserves: don't, it's, they're (apostrophe between letters)
        #    Removes:   'hello' (leading), dogs' (trailing)
        py_str = re.sub(r"(?<!\w)'|'(?!\w)", " ", py_str)

        # 7. Collapse multiple spaces/newlines → single space, strip ends
        py_str = re.sub(r"\s+", " ", py_str).strip()

        # 8. Split and filter — drop words shorter than 2 characters.
        #    Single-char tokens ('a', 'i', 's') create noisy n-grams
        #    like 'a_film_i' that inflate vocab without semantic value.
        var filter_fn = Python.evaluate(
            "lambda words: [w for w in words.split() if len(w) >= 2]"
        )
        return filter_fn(py_str)

    fn content_words(self, words: PythonObject) raises -> PythonObject:
        """Filter stopwords from a cleaned word list for n-gram assembly.

        Called by SimpleTokenizer.ngramify before assembling bigrams
        and trigrams. Unigrams always use the full cleaned word list —
        stopword filtering applies to n-grams only.

        Removing stopwords from n-gram construction eliminates noisy
        combinations like 'on_it_it', 'is_the_film', 'a_good' that
        pass frequency thresholds but carry no sentiment signal.
        Meaningful n-grams like 'not_good', 'highly_recommend',
        'waste_of_time' are preserved because their component words
        are not stopwords.

        Args:
            words: Python list of cleaned word strings from clean_text().

        Returns:
            Python list with stopwords removed, ready for n-gram assembly.
        """
        var filter_fn = Python.evaluate(
            "lambda words, sw: [w for w in words if w not in sw]"
        )
        var stopwords = Python.evaluate(STOPWORDS)
        return filter_fn(words, stopwords)
