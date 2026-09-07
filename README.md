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

## Streamlit Community Cloud

Configure these secrets/settings after the final artifact is verified:

- `HF_TOKEN`: read access to the private model, if it remains private.
- `MODEL_REVISION`: optional override for the pinned verified commit.
- `MODEL_ID`: optional override for the default repository.

The checked-in default model revision is
`11215925b0f9b64cfcfbbb5492b52d6aeb5a572b`.

The application caches the validated model process-wide with
`st.cache_resource`. DeBERTa-v3-large may exceed Community Cloud resource
limits; verify a cold start before sharing the link. If it does not fit, keep
this UI and host the same canonical predictor on a suitable CPU/GPU service.

## Safety

This is a model evaluation interface, not the ScanCode review/apply command.
Actual rule changes must use the validation, preview, human-review, and apply
workflow developed in ScanCode Toolkit PRs #5262 and #5267.

Licensed under the Apache License 2.0. See `LICENSE`.
