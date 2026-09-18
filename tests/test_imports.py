"""The installed-path Molvid package must not acquire legacy root imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_flat_package_is_importable_outside_repository_without_legacy_roots(tmp_path: Path):
    script = """
import importlib
import inspect
import json
import pkgutil
import sys

import molvid

names = sorted(item.name for item in pkgutil.walk_packages(molvid.__path__, "molvid."))
for name in names:
    importlib.import_module(name)
legacy = sorted(
    root for root in ("module", "trainer", "evaluation", "utils", "scripts", "data")
    if root in sys.modules or any(name.startswith(root + ".") for name in sys.modules)
)
old_public = sorted(
    f"{name}.{symbol}"
    for name in names
    for symbol, value in vars(sys.modules[name]).items()
    if inspect.isclass(value) and value.__module__ == name and symbol.startswith("PVB")
)
print(json.dumps({"modules": names, "legacy": legacy, "old_public": old_public}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    loaded = json.loads(result.stdout)
    assert "molvid.cli.sample" in loaded["modules"]
    assert "molvid.cli.evaluate" in loaded["modules"]
    assert "molvid.checkpoints" in loaded["modules"]
    assert loaded["legacy"] == []
    assert loaded["old_public"] == []
    assert not (ROOT / "src").exists()
    assert not (ROOT / "molvid" / "models").exists()


def test_package_discovery_is_scoped_to_molvid_only():
    from setuptools import find_packages

    packages = find_packages(where=str(ROOT), include=["molvid", "molvid.*"])
    assert "molvid" in packages
    assert all(name == "molvid" or name.startswith("molvid.") for name in packages)
    assert "molvid.models" not in packages
