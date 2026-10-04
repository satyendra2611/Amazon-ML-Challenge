# Business Entity Resolution — reproducible pipeline

End-to-end solution for the ML Challenge 2026 Business Entity Resolution task:
for every Source-1 entity, find all matching Source-2 / Source-3 records.

**Submitted model:** `models/matcher_v3.joblib` — HistGradientBoostingClassifier over
57 engineered similarity features, fed by country-gated IDF sparse retrieval with
ALIAS blocking, decided by a threshold of tau = 0.70.

**Held-out validation macro F0.5 = 0.956248** (60,000 Source-1 entities never used for
training; macro precision 0.979818, macro recall 0.905135, candidate recall 0.961853,
singleton accuracy 0.960867).  The test ground truth is not available locally, so this
is a validation figure and **not** a test or leaderboard score.

### Measured improvement experiment

The validation blocking audit found 8,793 missed true pairs among 208,091. Of
these, 1,815 were ranked within top 40 per source, compared with top 20 used in
the submitted pipeline. A wider candidate set can recover some of them, but it
also creates more hard negatives and can lower precision. Run the controlled
experiment after steps 1-5b above:

```bash
python3 -m src.experiment_k40
```

This generates top-40 train/validation candidates, samples 100,000 complete
Source-1 candidate groups for memory-bounded fitting, retrains the matcher, and
independently verifies macro F0.5. It leaves the submitted model and outputs
untouched. If the printed score beats the shipped 0.956248 by at least 0.0005,
generate new test outputs with:

```bash
python3 -m src.experiment_k40 --produce-test
```

On the supplied dataset, this experiment produced **0.955419** held-out macro
F0.5 with candidate recall **0.970484**. The submitted top-20 model remains
better at **0.956248**, so the final package retains its original model and
output TSVs. The original matcher scored **0.953358** when evaluated on the
top-40 candidates, confirming that wider blocking alone did not improve the
decision outcome.

The command audits both output files and runs the supplied validator. It needs
the seven original TSVs under `dataset/` (or `BER_DATA_DIR`). Do not claim a
0.99 score without a scored leaderboard upload; the held-out US/India validation
set cannot measure performance on test-only France. The existing token list was
discovered using the full training ground truth, including validation entities,
so a final rigorous check should rebuild a train-only token list, normalize again,
and repeat the experiment before choosing a submission.

Everything runs **offline** on the provided dataset. No external database, API,
geocoder, gazetteer, business registry, pretrained model download, or internet
access of any kind is used at any point — see *Fair play* at the bottom.

---

## 1. Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ (developed and measured on 3.14.6). All dependencies are pure PyPI
packages under BSD-3-Clause / MIT / Apache-2.0 — see `requirements.txt` for the
per-package licence list.

Hardware the reported numbers were produced on: **8-core CPU, 16 GB RAM, 4 GB swap,
macOS**.  No GPU is required or used.  Peak resident memory of the heaviest stage is
**6.9 GB** (model training), so **run one stage at a time** -- two concurrent stages
will thrash swap on a 16 GB machine.  Free disk needed for intermediates is ~4 GB.

Measured runtimes and peak RSS, one stage at a time:

| stage | runtime | peak RSS |
| --- | --- | --- |
| profile raw data | 194 s | 3.2 GB |
| normalize + cache 7 files | 122 s | 4.0 GB |
| candidate generation, test (69.3M pairs) | 2,696 s | 6.1 GB |
| model training (10M x 57) | 722 s | 6.9 GB |
| test inference (69.3M pairs) | 1,872 s | 4.7 GB |
| output audit | 115 s | 0.6 GB |

## 2. Data placement

Lay the competition data out exactly as shipped, relative to this folder:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
utils/validate_submission.py      # official validator (shipped with the challenge)
```

Any other location works by exporting `BER_DATA_DIR=/path/to/dataset`; all paths
come from `src/config.py` and nothing is hard-coded to a personal machine.

> The TSVs must be read with an explicit tab separator **and with quoting
> disabled** — the data contains bare `"` characters inside business names (349 rows
> in `test_source2.tsv` alone). `src/io_utils.py` does this; do not substitute a
> plain `pd.read_csv(..., sep="\t")`.

## 3. Reproduce end to end

Run from this directory, in order. Wall-clock times are measured on the machine
above; each step caches its output, so steps can be re-run independently.

```bash
# 0. sanity: the metric implementation reproduces the worked example in the spec
python3 -m src.evaluate                                        # ~0s

# 1. profile the raw data (row counts, nulls, countries, script mix, GT stats)
python3 -m src.profile_data                                    # ~3 min

# 2. normalize all 7 files once, cached as parquet
python3 -m src.prepare                                         # ~2 min

# 3. discover the legal-suffix / noise-token list from the ground truth
python3 -m src.discover_tokens                                 # ~10 s

# 4. deterministic Source-1-level train/validation split
python3 -m src.make_split                                      # ~20 s

# 5. corpus IDF tables (one per split)
python3 -m src.features train && python3 -m src.features test   # ~3 min

# 5b. learn the token-substitution alias map from the training ground truth
python3 -m src.discover_aliases                                # ~15 s

# 6. candidate generation (blocking) with ALIAS, for training and validation
python3 -m src.gen_candidates --split train --queries train --sample 250000 \
        --topk 20 --w-alias-name-tok 0.9 --tag trainsub_k20_alias   # ~11 min
python3 -m src.gen_candidates --split train --queries val \
        --topk 20 --w-alias-name-tok 0.9 --tag val_k20_alias        # ~5 min

# 7. train the matcher and choose the decision rule by validation macro F0.5
python3 -m src.train \
        --train-cand artifacts/cache/cand_train_trainsub_k20_alias.parquet \
        --val-cand   artifacts/cache/cand_train_val_k20_alias.parquet \
        --exp v3 --chunk 1000000 --max-iter 400 --lr 0.10 \
        --max-leaf 63 --min-leaf 100 --l2 1.0                       # ~12 min

# 7b. independently re-measure the saved model (metric recomputed from the
#     formula; does NOT import src/evaluate.py)
python3 -m src.verify_model --model models/matcher_v3.joblib \
        --cand artifacts/cache/cand_train_val_k20_alias.parquet --sweep   # ~2 min

# 8. error analysis + blocking failure audit (writes artifacts/errors/*.csv)
python3 -m src.error_analysis --model models/matcher_v3.joblib \
        --cand artifacts/cache/cand_train_val_k20_alias.parquet      # ~4 min
python3 -m src.audit_blocking                                        # ~7 min

# 8b. final pre-flight: model config, threshold, open-set country, fair play
python3 -m src.preflight                                             # ~1 min

# 9. candidate generation for the TEST set (this IS the candidate set the
#    matcher scores, and what candidate_pairs.tsv contains)
python3 -m src.gen_candidates --split test --queries all \
        --topk 20 --w-alias-name-tok 0.9 --tag all_k20_alias         # ~45 min

# 10. test inference -> output/matching_results.tsv + output/candidate_pairs.tsv
python3 -m src.infer --model models/matcher_v3.joblib \
        --cand artifacts/cache/cand_test_all_k20_alias.parquet \
        --chunk 2000000                                              # ~31 min

# 10b. independent output audit (streams both TSVs; proves tab-delimited,
#      one row per entity, ids exist, matches subset of candidates)
python3 -m src.audit_output                                          # ~2 min

# 11. official validator (must print PASS)
python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

# 12. build the submission zip (re-runs the validator and refuses if it fails)
python3 -m src.package_submission --team-name <team_name>
```

`run_all.sh` executes the whole sequence in one go, one stage at a time.

### Determinism of the final outputs

Re-running steps 9–10 regenerates `output/candidate_pairs.tsv` and
`output/matching_results.tsv` byte-for-byte.  Verified: emitting the candidate file
through `--candidates-only` and through the full scoring run produced the identical
md5 (`de11f8dad84dc2232b5b63326d4672a5`).  Both files are written from the *same*
candidate table in one pass, which is what makes "every match is also a candidate"
structural rather than merely asserted.

### Output locations

| File | Meaning |
| --- | --- |
| `output/matching_results.tsv` | final matches — the scored submission |
| `output/candidate_pairs.tsv` | the candidate set the model ran inference over |
| `models/matcher_v3.joblib` | **submitted** classifier + chosen decision rule |
| `models/PRODUCTION.md` | which model is production, and the evidence for it |
| `artifacts/profile/` | measured data profile, match anatomy, discovered tokens |
| `artifacts/validation/` | split definition, blocking sweeps, training report |
| `artifacts/errors/` | false-positive / false-negative dumps with all features |
| `artifacts/experiments.csv` | running experiment log |

## 4. Determinism

* Single seed (`SEED = 20260925` in `src/config.py`) drives the split, the
  negative/entity sampling and the classifier.
* Candidate retrieval breaks score ties by ascending candidate id
  (`np.lexsort((col, -score))`), so the candidate list does not depend on hash or
  dict iteration order.
* Output ID lists are sorted (Source-2 before Source-3, then numeric id) and
  de-duplicated before writing.
* Dependency versions are pinned in `requirements.txt`.

Re-running steps 6–10 reproduces byte-identical TSVs.

## 5. Module map

| File | Role |
| --- | --- |
| `src/config.py` | paths, seed, tunables |
| `src/io_utils.py` | TSV reading (tab-separated, **quoting disabled**), TSV writing |
| `src/preprocessing.py` | Unicode/Indic normalization, legal suffixes, addresses |
| `src/profile_data.py` | data profiling |
| `src/prepare.py` | normalize + cache all sources as parquet |
| `src/dataset.py` | prepared-split loading, eid↔row lookup |
| `src/discover_tokens.py` | derives the noise-token list from ground truth |
| `src/make_split.py` | Source-1-level stratified validation split |
| `src/evaluate.py` | competition macro F_0.5 + blocking metrics (self-tested) |
| `src/blocking.py` | IDF-weighted inverted index, country-gated retrieval |
| `src/gen_candidates.py` | persists the candidate set |
| `src/features.py` | pairwise features + corpus IDF |
| `src/build_features.py` | streams feature chunks, joins labels |
| `src/train.py` | trains the matcher, sweeps decision rules |
| `src/decide.py` | threshold / margin / expected-F_0.5 decision rules |
| `src/error_analysis.py` | FP/FN dumps and pattern buckets |
| `src/infer.py` | test inference, writes both output TSVs |
| `src/package_submission.py` | validates and builds the submission zip |
| `src/tune_blocking.py`, `src/sweep_blocking.py`, `src/diag_rank.py`, `src/eval_blocking.py` | blocking experiments and diagnostics |

## 6. Fair play

* The only inputs are the seven provided TSVs.
* `src/preprocessing.py` contains one non-obvious table: an Indic→Latin
  romanization map. It is **generated at import time from Python's own
  `unicodedata` character database** (`DEVANAGARI LETTER KA` → `k`), i.e. from
  Unicode character properties, not from any business, address or geographic data
  source. This is necessary because 26.9% of Source-2 business names are written in
  Indic scripts while Source 1 is 100% Latin; naive ASCII folding deletes them.
* The address abbreviation table (`Rd`↔`Road`, `Rue`, `Nagar`, …) and the
  legal-form token list are plain word lists; the legal-form list is additionally
  *derived from the training ground truth* by `src/discover_tokens.py`. Neither is
  a gazetteer and neither maps a name to a real-world entity.
* No geocoding, no postal database, no registry lookup, no external dataset, no
  API call, no LLM, no downloaded embeddings or model weights.
* The matcher is `sklearn.ensemble.HistGradientBoostingClassifier` (BSD-3-Clause),
  trained from scratch on the provided training data. Its parameter count is on the
  order of 10^5 — far below the 8-billion limit — and it satisfies the
  MIT/Apache-2.0-compatible licence requirement.
* The pipeline never reads the test set's labels (there are none) and no test
  record is inspected or hard-coded anywhere.
