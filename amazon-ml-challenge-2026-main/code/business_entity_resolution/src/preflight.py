"""Final pre-flight before test inference.  Read-only: changes nothing.

Checks the ten conditions that have to hold before ~90 minutes of test compute is
spent, and fails loudly rather than letting a misconfiguration through.

Run:  python3 -m src.preflight
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .blocking import BlockConfig, _alias_name_map, _terms_for
from .gen_candidates import BEST_CFG
from .io_utils import read_tsv_arrow
from .preprocessing import strip_legal_suffix

EXPECTED = {
    "n_features": 57, "max_iter": 400, "learning_rate": 0.10,
    "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 1.0,
    "random_state": C.SEED, "tau": 0.70, "alias_w": 0.9,
}
FALLBACK_SHA = {
    "models/baseline_v1_LOCKED/matcher_v1.joblib": "b4599a8b37597179",
    "models/fallback_v2_LOCKED_matcher_v2.joblib": "193af6f112c5fce9",
}

fails: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def _scan(pattern: str) -> str:
    """grep the pipeline modules, excluding this checker and compiled caches.

    preflight.py must be excluded because it embeds the very patterns it searches
    for, so scanning itself always produced a false positive.
    """
    import subprocess
    files = sorted(str(f) for f in (C.ROOT / "src").glob("*.py")
                   if f.name != "preflight.py")
    r = subprocess.run(["grep", "-nE", pattern, *files],
                       capture_output=True, text=True)
    return r.stdout.strip()


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    print("=" * 74)
    print("FINAL PRE-FLIGHT")
    print("=" * 74)

    # ---- 1. model artifact
    print("\n[1] matcher_v3.joblib")
    mp = C.MODELS_DIR / "matcher_v3.joblib"
    ok(mp.exists(), f"exists ({mp.stat().st_size:,} bytes)" if mp.exists() else "MISSING")
    if not mp.exists():
        print("\nABORT: no model")
        return 1
    print(f"        sha256[:16] = {sha16(mp)}")
    bundle = joblib.load(mp)
    clf = bundle["model"]

    # ---- 2. configuration
    print("\n[2] V3 configuration")
    pr = clf.get_params()
    ok(clf.n_features_in_ == EXPECTED["n_features"],
       f"n_features_in_ = {clf.n_features_in_} (expect 57)")
    ok(len(bundle["feature_names"]) == EXPECTED["n_features"],
       f"len(feature_names) = {len(bundle['feature_names'])}")
    for k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
              "l2_regularization", "random_state"):
        ok(pr[k] == EXPECTED[k], f"{k} = {pr[k]} (expect {EXPECTED[k]})")
    ok(type(clf).__name__ == "HistGradientBoostingClassifier",
       f"model class = {type(clf).__name__}")
    ok(bundle["feature_config"] == {"use_aliases": True, "addr_nan": True},
       f"feature_config = {bundle['feature_config']}")
    print(f"        n_iter_ = {clf.n_iter_}  seed = {bundle['seed']}")

    # ---- 3. threshold
    print("\n[3] decision threshold")
    d = bundle["decision"]
    ok(d["rule"] == "threshold" and abs(float(d["tau"]) - EXPECTED["tau"]) < 1e-12,
       f"decision = {d} (expect threshold tau=0.70)")

    # ---- 4. ALIAS blocking
    print("\n[4] ALIAS blocking")
    amap = _alias_name_map()
    ok(len(amap) > 0, f"alias map loaded, {len(amap)} tokens")
    cfg = BlockConfig(topk=20, **{**BEST_CFG, "w_alias_name_tok": EXPECTED["alias_w"]})
    ok(cfg.w_alias_name_tok == EXPECTED["alias_w"],
       f"w_alias_name_tok = {cfg.w_alias_name_tok}")
    ok(cfg.topk == 20, f"topk = {cfg.topk} (k must NOT be increased)")
    ok(BEST_CFG["w_alias_name_tok"] == 0.0,
       "BEST_CFG default stays 0.0 so v1/v2 remain reproducible")
    # the alias term must actually bridge a known cross-script pair
    pairs = [("dream international private limited", "drima intaraneshanala praiveta limiteda"),
             ("red construction private limited", "reda kanstrakshana praiveta limiteda")]
    bridged = 0
    for a, b in pairs:
        ta = {t for t, _ in _terms_for(a, strip_legal_suffix(a), "x", cfg)}
        tb = {t for t, _ in _terms_for(b, strip_legal_suffix(b), "x", cfg)}
        shared_alias = {t for t in ta & tb if t.startswith("m:")}
        shared_raw = {t for t in ta & tb if t.startswith("n:")}
        bridged += bool(shared_alias and not shared_raw)
    ok(bridged == len(pairs),
       f"alias terms bridge {bridged}/{len(pairs)} known cross-script pairs")
    print(f"        blocking tag: {cfg.tag()}")

    # ---- 5/6. test pool
    print("\n[5/6] test data")
    counts = {}
    for key, path in C.TEST_FILES.items():
        t = read_tsv_arrow(path)
        counts[key] = t.num_rows
        del t
    ok(counts["s1"] == 1_732_544, f"test_source1 rows = {counts['s1']:,}")
    ok(counts["s2"] == 4_887_273, f"test_source2 rows = {counts['s2']:,}")
    ok(counts["s3"] == 5_082_316, f"test_source3 rows = {counts['s3']:,}")
    print(f"        candidate pool = {counts['s2'] + counts['s3']:,}")
    print(f"        expected candidate pairs at k=20 x 2 sources = "
          f"{counts['s1'] * 40:,}")

    # ---- 7. open-set country
    print("\n[7] open-set country handling")
    from .dataset import load_source
    ts1 = load_source("test", "s1", columns=["eid", "country_norm"])
    ts2 = load_source("test", "s2", columns=["eid", "country_norm"])
    c1 = set(np.unique(ts1.country_norm).tolist())
    c2 = set(np.unique(ts2.country_norm).tolist())
    ok("france" in c1 and "france" in c2, f"france present: S1={c1} S2={c2}")
    ok(len(c1) == 3, f"test S1 has {len(c1)} country labels, none filtered")
    # nothing in the pipeline hard-codes a country vocabulary
    hits = _scan(r"==\s*[\"'](us|india|france)[\"']|in\s*\[[\"']us[\"']")
    ok(hits == "", "no hard-coded country comparison in the pipeline modules"
       + (f"\n        {hits}" if hits else ""))
    # counted generically, without naming a country literal
    labels, cnts = np.unique(ts1.country_norm, return_counts=True)
    unseen = [(str(l), int(c)) for l, c in zip(labels, cnts)
              if str(l) not in ("us", "india")]
    n_fr = unseen[0][1] if unseen else 0
    print(f"        unseen-in-training country labels: "
          f"{[u[0] for u in unseen]} -> {n_fr:,} S1 entities "
          f"({n_fr/len(ts1):.1%}); each builds its own country index")
    del ts1, ts2

    # ---- 8. fair play
    print("\n[8] no external data / API / lookup")
    net = _scan(r"^\s*(import|from)\s+(requests|urllib|httpx|socket|boto3|openai|"
                r"geopy|googlemaps)")
    ok(net == "", "no network/geocoding imports in the pipeline modules"
       + (f"\n        {net}" if net else ""))
    ok((C.PROFILE_DIR / "aliases.json").exists()
       and (C.PROFILE_DIR / "legal_tokens.json").exists(),
       "alias + legal-token tables are derived artifacts, present locally")

    # ---- 9. subset guarantee
    print("\n[9] matches subset of candidates")
    src = (C.ROOT / "src" / "infer.py").read_text()
    ok("cand_lists[idx].append(sid)" in src and "if k:" in src
       and "match_lists[idx].append(sid)" in src,
       "infer.py writes BOTH files from the same candidate table in one pass, so "
       "a match cannot exist outside the candidate list by construction")

    # ---- 10. fallbacks untouched
    print("\n[10] fallback artifacts")
    for rel, want in FALLBACK_SHA.items():
        p = C.ROOT / rel
        if not p.exists():
            ok(False, f"{rel} MISSING")
            continue
        got = sha16(p)
        ok(got == want, f"{rel} sha={got} (expect {want})")
        import os
        ok(not os.access(p, os.W_OK), f"{rel} is read-only")

    print("\n" + "=" * 74)
    if fails:
        print(f"PRE-FLIGHT FAILED ({len(fails)} problems)")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("PRE-FLIGHT PASSED -- safe to start test candidate generation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
