"""Thin adapter from Streamlit to scancode-required-phrases inference."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from scancode_required_phrases.inference import RequiredPhrasePredictor


DEFAULT_MODEL_ID = (
    "Kaushik-Kumar-CEG/scancode-required-phrases-deberta-bioes-crf-hardened"
)
DEFAULT_MODEL_REVISION = "11215925b0f9b64cfcfbbb5492b52d6aeb5a572b"


def resolve_model_dir(model_id=None, token=None):
    """Download one immutable model snapshot and return its model files only."""
    model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    local_path = Path(model_id)
    if local_path.is_dir():
        return local_path

    revision = os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
        )
    )
    marker = json.loads((snapshot / "SUCCESS.json").read_text(encoding="utf-8"))
    names = ["SUCCESS.json", *marker["files"]]
    if any(len(Path(name).parts) != 1 for name in names):
        raise ValueError("Success_Marker contains an invalid artifact path")

    target = Path.home() / ".cache" / "scancode-required-phrases" / revision
    if target.is_dir() and {path.name for path in target.iterdir()} == set(names):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{revision}.", dir=target.parent))
    try:
        for name in names:
            source = snapshot / name
            try:
                os.link(source, staging / name)
            except OSError:
                shutil.copy2(source, staging / name)
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_predictor(model_id=None, token=None):
    """Load only a complete, validated hardened final model."""
    model_dir = resolve_model_dir(model_id=model_id, token=token)
    return RequiredPhrasePredictor.from_model_dir(model_dir), model_dir


def model_summary(model_dir):
    """Return stable provenance fields suitable for the public UI."""
    manifest = json.loads(
        (Path(model_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )
    identity = manifest["model_identity"]
    return {
        "base_model": identity["name"],
        "base_revision": identity["resolved_backbone_revision"],
        "selected_checkpoint": manifest["selected_checkpoint"],
        "validation_f1": manifest["best_validation_f1"],
    }
