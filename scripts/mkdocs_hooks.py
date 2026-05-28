"""MkDocs hooks for the CX Complaint Intelligence docs site.

Copies the executed notebooks from ``notebooks/`` into ``docs/notebooks/`` so
mkdocs-jupyter can render them as part of the site.  The destination is
gitignored &mdash; the notebooks remain canonical under the top-level
``notebooks/`` directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_SOURCE = PROJECT_ROOT / "notebooks"
NOTEBOOK_TARGET = PROJECT_ROOT / "docs" / "notebooks"


def on_pre_build(config: MkDocsConfig) -> None:
    """Mirror ``notebooks/`` into ``docs/notebooks/`` before the build runs.

    The target directory is wiped first so removed notebooks do not linger in
    successive builds.
    """
    if NOTEBOOK_TARGET.exists():
        shutil.rmtree(NOTEBOOK_TARGET)
    NOTEBOOK_TARGET.mkdir(parents=True)

    for ipynb in sorted(NOTEBOOK_SOURCE.glob("*.ipynb")):
        shutil.copy2(ipynb, NOTEBOOK_TARGET / ipynb.name)

    readme = NOTEBOOK_SOURCE / "README.md"
    if readme.exists():
        shutil.copy2(readme, NOTEBOOK_TARGET / "index.md")
