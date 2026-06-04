from __future__ import annotations

import pathlib

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_project_metadata_declares_alpha_release() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "vla-fastvlm"
    assert metadata["project"]["version"] == "0.1.0a0"
    assert "lerobot" in metadata["project"]["keywords"]
    assert "fastvlm" in metadata["project"]["keywords"]


def test_readme_explains_oss_status_and_plan() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for heading in ("Why this matters", "Roadmap", "Known limitations"):
        assert f"## {heading}" in readme


def test_core_package_layout_exists() -> None:
    package_root = ROOT / "src" / "vla_fastvlm"

    assert (package_root / "fastvla").is_dir()
    assert (package_root / "lerobot_fastvla").is_dir()
    assert (package_root / "model").is_dir()
