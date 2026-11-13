import pytest

from detecteval import textprep
from detecteval.textprep import normalise, strip_markdown, trim_words


def test_short_text_is_rejected():
    assert trim_words("too short", max_words=10, min_words=5) is None
    assert trim_words("   ", max_words=10, min_words=1) is None


def test_text_within_limit_is_only_normalised():
    assert trim_words("one  two\nthree four five", max_words=10, min_words=5) == "one two three four five"


def test_cut_falls_back_to_sentence_end():
    text = "A b c d e. F g h i j. K l m n o p q r"
    assert trim_words(text, max_words=12, min_words=5) == "A b c d e. F g h i j."


def test_hard_cut_when_no_sentence_end_in_window():
    text = " ".join(str(i) for i in range(30))
    assert trim_words(text, max_words=10, min_words=5) == " ".join(str(i) for i in range(10))


def test_sentence_end_before_min_words_is_ignored():
    text = "A b. " + " ".join("w" for _ in range(30))
    assert len(trim_words(text, max_words=10, min_words=5).split()) == 10


def test_invalid_limits():
    with pytest.raises(ValueError):
        trim_words("a b c", max_words=2, min_words=5)


def test_strip_markdown_and_normalise():
    assert normalise(strip_markdown("# Title\n**Bold** text\n- item `code`")) == "Title Bold text item code"


def test_fill_placeholders():
    from detecteval.build_dataset import fill_placeholders

    assert fill_placeholders("Dear @CAPS1, in @LOCATION2 @NUM1 people use @UNKNOWN3.") == \
        "Dear Local, in Ohio ten people use it."
    assert fill_placeholders("me@example.com stays") == "me@example.com stays"


def test_strip_preamble():
    from detecteval.textprep import strip_preamble

    assert strip_preamble("Here is the rewritten text:\nActual body.") == "Actual body."
    assert strip_preamble("Here the river bends.\nSecond line.") == "Here the river bends.\nSecond line."


def test_rung_text_needs_most_of_the_rung():
    text = " ".join(f"w{n}." for n in range(100))
    assert len(textprep.rung_text(text, 75).split()) == 75
    assert textprep.rung_text(text, 150) is None          # 100 words cannot stand for a 150-word rung
    assert len(textprep.rung_text(text, 110).split()) == 100


def test_html_to_text_drops_code_and_quotes():
    body = "<p>I tried &amp; failed.</p><pre><code>x = 1</code></pre><blockquote><p>quoted</p></blockquote><p>Why?</p>"
    assert textprep.normalise(textprep.html_to_text(body)) == "I tried & failed. Why?"
