import pytest

from detecteval import detector
from detecteval.calibration import share_at_least, share_at_most


def words(n: int, per_sentence: int = 10) -> str:
    return " ".join(f"word{i}." if (i + 1) % per_sentence == 0 else f"word{i}" for i in range(n))


CALIBRATION = {
    "public": {"classifier-mirrors": {str(rung): {"human": [float(v) for v in range(100)],
                                                  "ai": [float(v) for v in range(50, 150)]}
                                      for rung in (75, 150, 300, 500)}},
    "private": None,
}


def test_reference_shares():
    reference = [1.0, 2.0, 2.0, 5.0]
    assert share_at_least(reference, 2.0) == 0.75
    assert share_at_least(reference, 6.0) == 0.0
    assert share_at_most(reference, 2.0) == 0.75
    assert share_at_least([], 1.0) is None


def test_short_text_gives_no_passages():
    passages, truncated = detector.split_passages(words(40))
    assert passages == [] and truncated is False


def test_passages_take_the_longest_length_they_fill():
    passages, truncated = detector.split_passages(words(700))
    assert [p["rung"] for p in passages] == [500, 150]          # 500 words, then the 200 left over
    assert [p["words"] for p in passages] == [500, 200]
    assert len(passages[1]["text"].split()) == 150
    assert truncated is False


def test_long_text_is_capped_and_says_so():
    passages, truncated = detector.split_passages(words(500 * (detector.MAX_PASSAGES + 2)))
    assert len(passages) == detector.MAX_PASSAGES and truncated is True


@pytest.mark.parametrize(("score", "band"), [(99.5, "strong"), (96.0, "moderate"), (85.0, "weak"), (40.0, "none")])
def test_bands_follow_the_share_of_human_texts_scoring_as_high(score, band):
    assert detector.interpret("classifier-mirrors", 300, score, CALIBRATION)["band"] == band


def test_the_stricter_reference_decides_the_band():
    calibration = {**CALIBRATION, "private": {"classifier-mirrors": {"300": {"human": [98.0, 99.0, 99.0, 99.8]}}}}
    reading = detector.interpret("classifier-mirrors", 300, 99.5, calibration)
    assert reading["human_texts_scoring_at_least_this"] == {"public": 0.0, "private": 0.25}
    assert reading["band"] == "none"                             # a quarter of the private texts score this high


def test_result_never_names_a_verdict():
    passages, truncated = detector.split_passages(words(320))
    result = detector.build_result(passages, truncated, {"classifier-mirrors": [99.5], "binoculars": None},
                                   CALIBRATION)
    assert result["ok"] and result["summary"]["bands"]["strong"] == 1
    assert "binoculars" not in result["passages"][0]["detectors"]
    text = str({key: value for key, value in result.items() if key != "limits"}).lower()
    assert "written by ai" not in text and "is human" not in text       # labels describe the score, not the author
    assert result["summary"]["label"] == "Strongly AI-like"


@pytest.mark.parametrize(("share", "odds"), [(0.5, 2), (0.034, 30), (0.005, 200), (0.11, 10), (1.0, 2)])
def test_one_in_gives_round_odds(share, odds):
    assert detector.one_in(share) == odds


def test_plain_reading_says_how_common_the_score_is():
    assert detector.plain_reading("strong", 0.004, 500, 0.4) == "Only about 1 in 250 human texts score this high."
    assert detector.plain_reading("weak", 0.12, 500, 0.4) == "About 1 in 8 human texts score this high."
    assert detector.plain_reading("strong", 0.0, 260, 0.4) == "None of the 260 human reference texts scored this high."
    human_like = detector.plain_reading("none", 0.6, 500, 0.1)
    assert human_like.startswith("More than half") and "does not rule AI out" in human_like
    assert detector.plain_reading("none", 0.6, 500, 0.005).endswith("Few AI texts score this low (about 1 in 200).")


def reading(band):
    return {"detectors": {"classifier-mirrors": {"band": band, "label": detector.LABELS[band], "plain": "x"}}}


@pytest.mark.parametrize(("bands", "label"), [
    (["strong"], "Strongly AI-like"), (["strong", "strong"], "Strongly AI-like"), (["strong", "moderate"], "AI-like"),
    (["strong", "none"], "Mixed"), (["weak", "none"], "Mostly human-like"), (["none", "none"], "Human-like")])
def test_summary_label_for_the_whole_text(bands, label):
    assert detector.summarise([reading(band) for band in bands])["label"] == label


def test_summary_sentence_agrees_in_number():
    def sentence(*bands):
        return detector.summarise([reading(band) for band in bands])["reading"]

    assert sentence("strong", "none", "weak").startswith("1 of 3 passages reads")
    assert sentence("strong", "moderate", "none").startswith("2 of 3 passages read ")
