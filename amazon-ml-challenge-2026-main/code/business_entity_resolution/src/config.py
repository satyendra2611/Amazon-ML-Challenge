"""Central configuration: paths, seeds, and tunables.

Everything the pipeline needs is derived from DATA_DIR, so the code is portable to
any machine that has the competition dataset laid out as shipped.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# Repo root = parent of src/.  Allows `python -m src.xxx` from the project root.
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "dataset"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

ARTIFACTS = Path(os.environ.get("BER_ARTIFACTS", ROOT / "artifacts"))
PROFILE_DIR = ARTIFACTS / "profile"
VALIDATION_DIR = ARTIFACTS / "validation"
CACHE_DIR = ARTIFACTS / "cache"
ERRORS_DIR = ARTIFACTS / "errors"
EXPERIMENTS_CSV = ARTIFACTS / "experiments.csv"

MODELS_DIR = Path(os.environ.get("BER_MODELS", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT", ROOT / "output"))
LOGS_DIR = ROOT / "logs"

for _d in (
    PROFILE_DIR, VALIDATION_DIR, CACHE_DIR, ERRORS_DIR,
    MODELS_DIR, OUTPUT_DIR, LOGS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = {
    "s1": TRAIN_DIR / "train_source1.tsv",
    "s2": TRAIN_DIR / "train_source2.tsv",
    "s3": TRAIN_DIR / "train_source3.tsv",
    "gt": TRAIN_DIR / "train_ground_truth.tsv",
}
TEST_FILES = {
    "s1": TEST_DIR / "test_source1.tsv",
    "s2": TEST_DIR / "test_source2.tsv",
    "s3": TEST_DIR / "test_source3.tsv",
}

# ---------------------------------------------------------------- determinism
SEED = 20260925

# ---------------------------------------------------------------- validation split
# Fraction of *training* Source-1 entities held out for validation.  The split is at
# the Source-1-entity level so no S1 entity contributes candidate pairs to both sides.
VAL_FRACTION = 0.06

# Sub-sample of the validation entities actually scored end-to-end.  Full validation
# is 130k+ S1 entities; scoring a deterministic sample keeps iteration fast while
# staying large enough for a tight confidence interval on macro F0.5.
VAL_SCORE_SAMPLE = 60_000

# ---------------------------------------------------------------- blocking
BETA = 0.5  # F-beta used by the competition

# Token is "generic" (low information) if it occurs in more than this fraction of
# records on the candidate side; generic tokens are not used as sole blocking keys.
MAX_TOKEN_DF_FRACTION = 0.002

# Hard cap on postings scanned per query token, to keep worst-case bounded.
MAX_POSTINGS_PER_TOKEN = 4_000

# Maximum candidates retained per (S1 entity, source) after cheap prefilter scoring.
TOPK_PER_SOURCE = 30

BATCH_S1 = 20_000  # S1 entities processed per blocking batch

__all__ = [n for n in dir() if not n.startswith("_")]
