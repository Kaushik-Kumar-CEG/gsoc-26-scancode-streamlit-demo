from types import SimpleNamespace

from presentation import confidence_label
from presentation import highlighted_tokens
from presentation import review_diff


def phrase(start, end):
    return SimpleNamespace(start_word=start, end_word=end)


def test_confidence_labels_do_not_imply_approval():
    assert confidence_label(0.95) == "Higher confidence"
    assert confidence_label(0.75) == "Needs close review"
    assert confidence_label(0.20) == "Low confidence"


def test_highlights_all_candidate_spans():
    rendered = highlighted_tokens(
        ("Use", "MIT", "License", "or", "Apache", "License"),
        (phrase(1, 2), phrase(4, 5)),
    )

    assert rendered == (
        "Use <mark>MIT License</mark> or <mark>Apache License</mark>"
    )


def test_review_diff_marks_all_candidates():
    diff = review_diff(
        ("Use", "MIT", "License", "or", "Apache", "License"),
        (phrase(1, 2), phrase(4, 5)),
    )

    assert "-Use MIT License or Apache License" in diff
    assert "+Use {{MIT License}} or {{Apache License}}" in diff


def test_escapes_untrusted_rule_text():
    rendered = highlighted_tokens(("<script>", "MIT"), (phrase(1, 1),))

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
