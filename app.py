import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from predictor import DEFAULT_MODEL_ID
from predictor import load_predictor
from predictor import model_summary
from presentation import confidence_label
from presentation import highlighted_tokens


st.set_page_config(
    page_title="ScanCode required-phrase model",
    page_icon="🔎",
    layout="centered",
)

EXAMPLES = {
    "MIT reference": "This software is released under the MIT License.",
    "SPDX tag": "SPDX-License-Identifier: LGPL-2.0-or-later",
    "LGPL notice": (
        "This library is free software; you can redistribute it and/or modify "
        "it under the terms of the GNU Lesser General Public License as "
        "published by the Free Software Foundation; either version 2.1 of the "
        "License, or (at your option) any later version."
    ),
}


def secret(name):
    """Read an optional Streamlit secret without requiring a secrets file."""
    try:
        return st.secrets.get(name)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading and validating the model…")
def cached_predictor(model_id, revision, token):
    if revision:
        os.environ["MODEL_REVISION"] = revision
    return load_predictor(model_id=model_id, token=token)


def show_model_details(summary, model_id):
    with st.expander("Model details"):
        st.write(f"**Artifact:** `{model_id}`")
        st.write(f"**Base model:** `{summary['base_model']}`")
        st.write(f"**Base revision:** `{summary['base_revision']}`")
        st.write(f"**Selected checkpoint:** `{summary['selected_checkpoint']}`")
        st.write(f"**Best validation span F1:** `{summary['validation_f1']:.4f}`")
        st.caption("The complete artifact is hash-validated before it is loaded.")


def main():
    st.title("Required-phrase candidate finder")
    st.write(
        "Test the GSoC 2026 model on ScanCode license-rule text. "
        "The demo is read-only and does not modify rule files."
    )
    st.warning(
        "Predictions are candidates for human review, not approved annotations. "
        "A wrong required phrase can suppress a valid license detection."
    )

    selected = st.selectbox("Example", ["Custom text", *EXAMPLES])
    if selected != "Custom text" and st.session_state.get("example") != selected:
        st.session_state["rule_text"] = EXAMPLES[selected]
        st.session_state["example"] = selected

    text = st.text_area(
        "Rule text",
        key="rule_text",
        height=220,
        placeholder="Paste the plain text body of a ScanCode .RULE file",
    )
    st.caption("Paste rule text only; omit YAML frontmatter and the `---` separator.")

    if not st.button("Find candidate phrases", type="primary", use_container_width=True):
        return
    if not text.strip():
        st.info("Enter rule text to run the model.")
        return

    model_id = os.environ.get("MODEL_ID") or secret("MODEL_ID") or DEFAULT_MODEL_ID
    revision = os.environ.get("MODEL_REVISION") or secret("MODEL_REVISION")
    token = os.environ.get("HF_TOKEN") or secret("HF_TOKEN")

    try:
        predictor, model_dir = cached_predictor(model_id, revision, token)
        with st.spinner("Running inference…"):
            result = predictor.predict(text)
    except Exception:
        st.error(
            "The verified model could not be loaded or inference failed. "
            "Check the deployment configuration and try again."
        )
        return

    if result.truncated:
        st.warning(
            "This input exceeded the model limit. A candidate touching the cut-off "
            "was omitted. Test a shorter rule before reviewing the result."
        )

    st.subheader("Candidates")
    if not result.phrases:
        st.info("The model did not identify a required-phrase candidate.")
    else:
        for number, phrase in enumerate(result.phrases, 1):
            st.markdown(f"**{number}. `{phrase.text}`**")
            st.caption(
                f"{confidence_label(phrase.confidence)} · "
                f"model confidence {phrase.confidence:.1%} · "
                f"words {phrase.start_word + 1}–{phrase.end_word + 1}"
            )

        st.markdown("**Model token view**")
        st.markdown(
            highlighted_tokens(result.words, result.phrases),
            unsafe_allow_html=True,
        )
        st.caption(
            "Highlighting follows the exact normalized tokens seen by the model; "
            "it is not an injection preview."
        )

    try:
        show_model_details(model_summary(model_dir), model_id)
    except (KeyError, OSError, ValueError):
        st.caption(f"Model artifact: `{model_id}`")

    st.divider()
    st.caption(
        "This interface only displays model output. ScanCode validation and the "
        "review/apply workflow remain required before changing any rule."
    )


if __name__ == "__main__":
    main()
