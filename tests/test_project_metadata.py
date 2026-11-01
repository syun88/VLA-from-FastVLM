from __future__ import annotations

import pathlib

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_project_metadata_declares_mlx_research_release() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert metadata["project"]["name"] == "vla-fastvlm"
    assert metadata["project"]["version"] == "0.2.0a0"
    assert any(value.startswith("mlx-vlm") for value in dependencies)
    assert all(not value.startswith("torch") for value in dependencies)
    assert "mlx" in metadata["project"]["keywords"]


def test_readme_explains_research_status_and_limits() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for heading in ("Why this matters", "Roadmap", "Known limitations"):
        assert f"## {heading}" in readme
    assert "legacy/torch_v01/" in readme
    assert "simulation-only" in readme


def test_active_package_does_not_import_torch() -> None:
    import sys

    import vla_fastvlm  # noqa: F401

    assert "torch" not in sys.modules


def test_core_package_layout_exists() -> None:
    package_root = ROOT / "src" / "vla_fastvlm"

    assert (package_root / "fastvla" / "backbone.py").is_file()
    assert (package_root / "fastvla" / "modeling_fastvla.py").is_file()
    assert (package_root / "data" / "metaworld.py").is_file()
    assert (package_root / "evaluation" / "metaworld.py").is_file()
