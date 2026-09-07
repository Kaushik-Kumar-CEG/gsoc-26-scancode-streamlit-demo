import json
from pathlib import Path

import predictor


def test_remote_snapshot_excludes_repository_metadata(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "run_manifest.json").write_text("{}", encoding="utf-8")
    (snapshot / ".gitattributes").write_text("metadata", encoding="utf-8")
    (snapshot / "SUCCESS.json").write_text(
        json.dumps(
            {
                "files": {
                    "model.safetensors": "unused",
                    "run_manifest.json": "unused",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(predictor, "snapshot_download", lambda **kwargs: snapshot)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    model_dir = predictor.resolve_model_dir("owner/model")

    assert {path.name for path in model_dir.iterdir()} == {
        "SUCCESS.json",
        "model.safetensors",
        "run_manifest.json",
    }
