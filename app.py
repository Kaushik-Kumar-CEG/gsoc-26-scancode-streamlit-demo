import logging
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
    page_title="GSoC Required Phrases Demo",
    page_icon="🔎",
    layout="centered",
)

logger = logging.getLogger(__name__)


EXAMPLES = {
    "MIT notice": "This software is released under the MIT License.",
    "Apache notice": (
        "Licensed under the Apache License, Version 2.0. You may obtain a copy "
        "of the License at http://www.apache.org/licenses/LICENSE-2.0."
    ),
    "SPDX tag": "SPDX-License-Identifier: LGPL-2.0-or-later",
    "BSD reference": "This code is distributed under the BSD 3-Clause License.",
    "LGPL notice": (
        "This library is free software; you can redistribute it and/or modify it "
        "under the terms of the GNU Lesser General Public License as published "
        "by the Free Software Foundation; either version 2.1 of the License, or "
        "at your option any later version."
    ),
    "GPL notice": (
        "This program is free software; you can redistribute it and/or modify "
        "it under the terms of the GNU General Public License as published by "
        "the Free Software Foundation; either version 2 of the License, or any "
        "later version."
    ),
    "Dual license": (
        "This library may be used under the terms of the GNU Lesser General "
        "Public License version 2.1 or, at your option, under the terms of the "
        "GNU General Public License version 2."
    ),
    "Mozilla notice": (
        "The contents of this file are subject to the Mozilla Public License "
        "Version 1.1."
    ),
    "URL only": "https://opensource.org/licenses/MIT",
    "No clear phrase": (
        "The full license terms are available in the documentation supplied "
        "with this package."
    ),
}


def secret(name):
    """Read an optional Streamlit secret without requiring a secrets file."""
    try:
        return st.secrets.get(name)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading and validating the model...")
def cached_predictor(model_id, revision, token):
    if revision:
        os.environ["MODEL_REVISION"] = revision
    return load_predictor(model_id=model_id, token=token)


def select_example(label):
    st.session_state["rule_text"] = EXAMPLES[label]


def show_examples():
    st.markdown("#### Try an example")
    labels = list(EXAMPLES)
    for start in range(0, len(labels), 2):
        columns = st.columns(2)
        for column, label in zip(columns, labels[start : start + 2]):
            column.button(
                label,
                key=f"example_{label}",
                on_click=select_example,
                args=(label,),
                use_container_width=True,
            )


def show_model_details(summary, model_id):
    with st.expander("Model details"):
        st.write(f"**Artifact:** `{model_id}`")
        st.write(f"**Base model:** `{summary['base_model']}`")
        st.write(f"**Base revision:** `{summary['base_revision']}`")
        st.write(f"**Selected checkpoint:** `{summary['selected_checkpoint']}`")
        st.write(f"**Best validation span F1:** `{summary['validation_f1']:.4f}`")
        st.caption("The model files and recorded hashes are validated before loading.")


def main():
    st.caption("Google Summer of Code 2026 | AboutCode")
    st.title("GSoC Required Phrases Demo")
    st.write(
        "Paste a ScanCode license rule or choose an example to find text that "
        "may need required phrase markers."
    )

    show_examples()

    st.markdown("#### Rule text")
    text = st.text_area(
        "Rule text",
        key="rule_text",
        height=240,
        placeholder="Paste the plain text body of a ScanCode .RULE file",
        label_visibility="collapsed",
    )
    st.caption("Use the rule text only. Leave out the YAML header and separator.")

    if not st.button("Find required phrases", type="primary", use_container_width=True):
        return
    if not text.strip():
        st.info("Enter rule text or choose an example first.")
        return

    model_id = os.environ.get("MODEL_ID") or secret("MODEL_ID") or DEFAULT_MODEL_ID
    revision = os.environ.get("MODEL_REVISION") or secret("MODEL_REVISION")
    token = os.environ.get("HF_TOKEN") or secret("HF_TOKEN")

    try:
        predictor, model_dir = cached_predictor(model_id, revision, token)
        with st.spinner("Finding candidate phrases..."):
            result = predictor.predict(text)
    except Exception as error:
        logger.exception("Model loading or inference failed")
        st.error(
            "The model could not be loaded. Check the app configuration and try again. "
            f"Error type: {type(error).__name__}."
        )
        return

    if result.truncated:
        st.warning(
            "This text is longer than the model limit. A phrase at the end may "
            "not be included in the results."
        )

    st.markdown("### Suggested required phrases")
    if not result.phrases:
        st.info("No required phrase was suggested for this text.")
    else:
        for number, phrase in enumerate(result.phrases, 1):
            with st.container(border=True):
                st.markdown(f"**{number}. `{phrase.text}`**")
                st.caption(
                    f"{confidence_label(phrase.confidence)} | "
                    f"Confidence {phrase.confidence:.1%} | "
                    f"Words {phrase.start_word + 1} to {phrase.end_word + 1}"
                )

        st.markdown("#### Highlighted result")
        st.markdown(
            highlighted_tokens(result.words, result.phrases),
            unsafe_allow_html=True,
        )

    try:
        show_model_details(model_summary(model_dir), model_id)
    except (KeyError, OSError, ValueError):
        st.caption(f"Model artifact: `{model_id}`")

    st.caption("Model suggestions should be reviewed before updating a ScanCode rule.")


if __name__ == "__main__":
    main()
