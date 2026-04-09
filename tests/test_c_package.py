"""Package-local tests for the first-party C analyzer distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path

from repoindex_analyzer_c import CAnalyzer, build_analyzer


def test_c_package_declares_expected_entry_point() -> None:
    """Keep package metadata aligned to the analyzer entry-point contract."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert project["project"]["entry-points"]["repoindex.analyzers"] == {
        "c": "repoindex_analyzer_c:build_analyzer"
    }


def test_c_package_builds_expected_analyzer() -> None:
    """Keep the package-local factory aligned to the published analyzer name."""
    analyzer = build_analyzer()

    assert isinstance(analyzer, CAnalyzer)
    assert analyzer.name == "c"
