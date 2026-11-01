from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_text_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    paths = []
    for relative in output.splitlines():
        path = ROOT / relative
        if not path.is_file() or path.suffix in {".png", ".jpg", ".mp4", ".safetensors"}:
            continue
        paths.append(path)
    return paths


def test_public_tree_has_no_private_paths_or_wandb_identity() -> None:
    patterns = {
        "private home path": re.compile(r"/Users/[^/\s]+"),
        "external-volume path": re.compile(r"/Volumes/[^/\s]+"),
        "named W&B run URL": re.compile(r"https://wandb\.ai/[A-Za-z0-9_-]+/"),
        "hard-coded credential": re.compile(
            r"(?:WANDB_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*[^\s<]+"
        ),
    }
    findings = []
    for path in _tracked_text_files():
        if "tests/test_public_repository_hygiene.py" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    assert not findings, "\n".join(findings)


def test_no_model_weights_or_credentials_are_tracked() -> None:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    forbidden = (
        ".safetensors",
        ".pem",
        ".key",
        ".p12",
        ".netrc",
        "credentials.json",
    )
    assert not [path for path in output.splitlines() if path.endswith(forbidden)]
