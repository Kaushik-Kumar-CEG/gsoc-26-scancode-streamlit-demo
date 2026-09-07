"""Presentation helpers for the required-phrase demo."""

import difflib
import html


CONFIDENCE_LABELS = (
    (0.95, "Higher confidence"),
    (0.60, "Needs close review"),
    (0.00, "Low confidence"),
)


def confidence_label(confidence):
    """Return a descriptive, non-approval confidence label."""
    for threshold, label in CONFIDENCE_LABELS:
        if confidence >= threshold:
            return label
    raise AssertionError("confidence thresholds must cover zero")


def marker_preview(words, phrases):
    """Return normalized model tokens with candidate markers inserted."""
    starts = {phrase.start_word for phrase in phrases}
    ends = {phrase.end_word for phrase in phrases}
    marked = []
    for index, word in enumerate(words):
        prefix = "{{" if index in starts else ""
        suffix = "}}" if index in ends else ""
        marked.append(f"{prefix}{word}{suffix}")
    return " ".join(marked)


def review_diff(words, phrases):
    """Return a review-style diff of the normalized model tokens."""
    before = " ".join(words) + "\n"
    after = marker_preview(words, phrases) + "\n"
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="rule text",
            tofile="candidate preview",
        )
    )


def highlighted_tokens(words, phrases):
    """Render escaped model tokens with candidate spans highlighted."""
    covered = set()
    for phrase in phrases:
        covered.update(range(phrase.start_word, phrase.end_word + 1))

    rendered = []
    inside = False
    for index, word in enumerate(words):
        selected = index in covered
        if selected and not inside:
            rendered.append("<mark>")
        if not selected and inside:
            rendered.append("</mark>")
        rendered.append(html.escape(word))
        inside = selected
    if inside:
        rendered.append("</mark>")
    return " ".join(rendered).replace("<mark> ", "<mark>").replace(" </mark>", "</mark>")
