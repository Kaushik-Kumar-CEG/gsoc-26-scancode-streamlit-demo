# ScanCode required-phrase model demo

A small, read-only Streamlit interface for maintainers to test the GSoC 2026
required-phrase model on license-rule text.

The application displays candidate phrase spans and model confidence. It does
not edit ScanCode rules or approve predictions. Every candidate still requires
human review because an incorrect required phrase can suppress a valid license
detection.

## Model contract

The demo deliberately contains no independent inference implementation. It uses
`scancode_required_phrases.inference.RequiredPhrasePredictor`, which:

- validates `SUCCESS.json` and every artifact hash;
- loads the hardened BIOES and constrained-CRF model offline;
- uses the same word tokenization and decoding contract as training; and
- returns candidates without changing files.

The default model repository is:

`Kaushik-Kumar-CEG/scancode-required-phrases-deberta-bioes-crf-hardened`

The final model is still training. Until it has been verified and uploaded, UI
and unit tests can run but live inference is expected to report that the model
cannot be loaded.

## Local development

Use Python 3.10 or newer. Install the hardened package source and then the demo:

```bash
python -m pip install -e ../scancode-required-phrases[training]
python -m pip install streamlit==1.54.0
streamlit run app.py
```

For a local final model directory:

```bash
MODEL_ID=/path/to/final-model streamlit run app.py
```

For the private Hugging Face repository, set `HF_TOKEN` in the environment. Do
not commit tokens or `.streamlit/secrets.toml`.

Run the lightweight UI tests with:

```bash
pytest -q test_presentation.py
```

## Hosted demo

The FP32 DeBERTa-v3-large model exceeds Streamlit Community Cloud's available
memory during loading. Deploy this repository as a Hugging Face Docker Space
instead. The included `Dockerfile` runs the same Streamlit application on port
7860 with Python 3.11.

Add `HF_TOKEN` as a private Space secret with read access to the model. Optional
`MODEL_ID` and `MODEL_REVISION` variables can override the checked-in defaults.
The default model revision is pinned to
`11215925b0f9b64cfcfbbb5492b52d6aeb5a572b`.

The application caches the validated model process-wide with
`st.cache_resource`.

## Safety

This is a model evaluation interface, not the ScanCode review/apply command.
Actual rule changes must use the validation, preview, human-review, and apply
workflow developed in ScanCode Toolkit PRs #5262 and #5267.

Licensed under the Apache License 2.0. See `LICENSE`.
