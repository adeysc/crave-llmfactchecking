"""Single source of truth for tcss_ad_crave data locations.

Everything lives under the repo root: the two K4 JSONs per dataset in `data/`, and all
generated artifacts (re-clusterings, judged files, experiment summaries) flat in
`outputs/`. Override either with an env var if you keep them elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# config/paths.py -> parents[1] == repo root
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CRAVE_DATA", str(ROOT / "data")))
OUTPUTS = Path(os.environ.get("CRAVE_OUTPUTS", str(ROOT / "outputs")))

OUTPUTS.mkdir(parents=True, exist_ok=True)
