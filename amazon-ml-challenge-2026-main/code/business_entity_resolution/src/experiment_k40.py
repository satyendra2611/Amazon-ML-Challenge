"""Run a measured top-40 candidate experiment; promote only on validation gain.

Run after the README's preparation, split, IDF and alias-discovery steps:
    python -m src.experiment_k40
    python -m src.experiment_k40 --produce-test

The second command regenerates the test output only if the independent validation
score exceeds the shipped v3 score by at least --min-gain. It never edits v3.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config as C

BASELINE = 0.956248005866283
TAG = "k40_alias"


def run(*args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print("[k40]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=C.ROOT, check=True)


def require(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing inputs; complete the README preparation steps first:\n"
                         + "\n".join(missing))


def metric() -> float:
    report = C.VALIDATION_DIR / "verify_matcher_k40.json"
    require([report])
    return float(json.loads(report.read_text())["stored_result"]["macro_f05"])


def fit_subset(source: Path, target: Path, n_entities: int) -> Path:
    """Keep complete candidate groups for a deterministic S1 sample."""
    if target.exists():
        return target
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    s1 = table["s1_eid"].to_numpy()
    unique = np.unique(s1)
    rng = np.random.default_rng(C.SEED + 40)
    chosen = np.sort(rng.choice(unique, size=min(n_entities, len(unique)),
                                replace=False))
    subset = table.filter(pa.array(np.isin(s1, chosen)))
    pq.write_table(subset, target, compression="zstd", compression_level=3)
    print(f"[k40] fit subset: {len(chosen):,} complete S1 groups, "
          f"{subset.num_rows:,} pairs -> {target}", flush=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--produce-test", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--train-sample", type=int, default=250_000)
    ap.add_argument("--fit-entities", type=int, default=100_000,
                    help="complete S1 groups used to fit the model; controls RAM")
    args = ap.parse_args()
    if args.min_gain < 0 or args.train_sample < 1 or args.fit_entities < 1:
        ap.error("--min-gain must be nonnegative; sample sizes must be positive")

    require([*C.TRAIN_FILES.values(),
             C.VALIDATION_DIR / "train_s1_eids.npy",
             C.VALIDATION_DIR / "val_score_s1_eids.npy",
             C.PROFILE_DIR / "aliases.json",
             C.CACHE_DIR / "train_s1.parquet",
             C.CACHE_DIR / "train_s2.parquet",
             C.CACHE_DIR / "train_s3.parquet"])

    val = C.CACHE_DIR / f"cand_train_val_{TAG}.parquet"
    train = C.CACHE_DIR / f"cand_train_trainsub_{TAG}.parquet"
    fit = C.CACHE_DIR / f"cand_train_fit{args.fit_entities}_{TAG}.parquet"
    model = C.MODELS_DIR / "matcher_k40.joblib"
    if not val.exists():
        run("src.gen_candidates", "--split", "train", "--queries", "val",
            "--topk", "40", "--w-alias-name-tok", "0.9", "--tag", f"val_{TAG}")
    if not train.exists():
        run("src.gen_candidates", "--split", "train", "--queries", "train",
            "--sample", str(args.train_sample), "--topk", "40",
            "--w-alias-name-tok", "0.9", "--tag", f"trainsub_{TAG}")
    fit_subset(train, fit, args.fit_entities)
    if not model.exists():
        run("src.train", "--train-cand", str(fit), "--val-cand", str(val),
            "--exp", "k40", "--chunk", "1000000", "--max-iter", "400",
            "--lr", "0.10", "--max-leaf", "63", "--min-leaf", "100",
            "--l2", "1.0")
    run("src.verify_model", "--model", str(model),
        "--cand", str(val), "--sweep")
    score = metric()
    gain = score - BASELINE
    print(f"[k40] independently verified F0.5={score:.6f}; "
          f"v3={BASELINE:.6f}; gain={gain:+.6f}", flush=True)
    if not args.produce_test:
        return 0
    if gain < args.min_gain:
        raise SystemExit(f"Test output not generated: gain {gain:+.6f} is below "
                         f"required +{args.min_gain:.6f}.")

    require(list(C.TEST_FILES.values()))
    test = C.CACHE_DIR / f"cand_test_all_{TAG}.parquet"
    if not test.exists():
        run("src.gen_candidates", "--split", "test", "--queries", "all",
            "--topk", "40", "--w-alias-name-tok", "0.9", "--tag", f"all_{TAG}")
    run("src.infer", "--model", str(model),
        "--cand", str(test), "--chunk", "2000000")
    run("src.audit_output")
    run("utils.validate_submission", "--matching", str(C.OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(C.OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(C.TEST_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
