# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Innovators]

**Team Members:**

1. Vikas Tiwari

**Submission Date:** 2026-09-25

---

## 1. Executive Summary

A country-gated, IDF-weighted sparse-retrieval blocker feeds a 57-feature gradient-boosted

pairwise classifier, with an F₀.₅-optimised per-entity accept threshold that is free to

return the empty set. The two ideas that mattered most were both forced by measurement

rather than assumed: **26.9% of Source-2 business names are written in Indic scripts and are

destroyed outright by naive ASCII folding**, so names are romanized from Unicode character

properties first; and **the systematic token substitutions between sources (`tx`↔`texas`,

`karnataka`↔`ka`, `energy`↔`enarji`) can be learned from the training ground truth**, which

lifted both blocking recall and the strongest single feature in the model.

**Held-out validation macro F₀.₅ = 0.956248 on 60,000 Source-1 entities excluded from pairwise model training.** The test ground truth is not available locally, so **no test/leaderboard score is

claimed anywhere in this document.**

---

## 2. Methodology

### 2.1 Problem Analysis

Everything below is measured (`artifacts/profile/profile.txt`, `match_anatomy.txt`), not assumed.

|                        |  train S1 |  train S2 |  train S3 |   test S1 |   test S2 |   test S3 |
| ---------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| rows                   | 2,206,821 | 5,034,616 | 5,285,603 | 1,732,544 | 4,887,273 | 5,082,316 |
| empty business_name    |         0 |         0 |         0 |         0 |         0 |         0 |
| empty business_address |         0 |   168,967 |   175,916 |         0 |   129,408 |   136,098 |

Ground truth: 2,206,821 S1 entities, **5.58% with zero matches**, mean 3.46 matches, max 11

(max 5 from S2, max 6 from S3). So ~94% of entities *do* have matches — recall matters much

more than a precision-weighted metric might suggest.

Five findings that shaped the design:

**1. Non-Latin names.** Source 1 is 100% Latin. Source 2/3 contain Devanagari, Telugu, Tamil,

Kannada, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi. `NFKD → encode("ascii","ignore")`

empties **1,353,239 of 5,034,616 (26.9%)** S2 names and 392,083 (7.4%) S3 names.

**2. Addresses are ~100% Latin** in every source (non-Latin < 0.06%), so address is the reliable

cross-script bridge.

**3. Country agrees on 100.00%** of 60,000 sampled true pairs — a safe hard blocking gate.

**4. Blocking ceiling.** 87.6% of true pairs share ≥1 name token, 95.6% share ≥1 address token,

**100.0% share one or the other.**

**5. Bare double-quote characters** appear in business names (349 rows in `test_source2.tsv`,

e.g. `"""chrs & cie sasu"`). A default `pd.read_csv(sep="\t")` treats them as quoting and

silently rewrites those names. Quoting is disabled everywhere (`src/io_utils.py`).

Noise mechanisms found by error analysis (`artifacts/errors/`):

* **de-spaced / domain-style names** — `ramirezinvestments com` vs `ramirez investments llc`,

  `interfaithcouncil` vs `interfaith council` (zero shared tokens);

* **truncated addresses** — `a 142 bldg no 5 and 6 kalpataru estate jv link rd majas vill`

  vs `a 142 mumbai maharashtra`;

* **fully synthetic replacement names** — `quonovi id 72868` at an identical address;

* **acronyms** — `ganpati consultancy private limited` → `gc`;

* **house-number disambiguation** — an identical name at `98 decatur st` vs `119 decatur st`

  is *not* a match, while a garbled name at the same number *is*.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pairwise classifier + per-entity decision layer.

**Core Innovation:** two data-derived normalization tables that need no external data —

(a) an Indic→Latin romanizer generated at import time from Python's own Unicode character

database, and (b) a token-substitution alias map learned from the training ground truth,

which recovers US *and* Indian state abbreviations and a romanized-Indic→English lexicon.

---

## 3. Candidate Generation (Blocking)

One retrieval pass, not seven unioned passes. Each record becomes a bag of IDF-weighted terms:

| term         | meaning                                          | weight |
| ------------ | ------------------------------------------------ | -----: |
| `n:<token>`  | `name_norm` token                                |    1.0 |
| `a:<token>`  | `addr_norm` token, street-type stopwords removed |    1.1 |
| `g:<4-gram>` | character 4-grams of the de-spaced `name_core`   |   0.35 |
| `m:<token>`  | **alias-canonicalized** name token               |    0.9 |

Retrieval is cosine similarity via one sparse matrix product against an inverted index

(terms × docs), then top-20 **per source** (40 candidates per entity). IDF weighting means

rare tokens dominate and generic ones (`private`, `limited`, `delhi`, `main`) contribute

almost nothing, so rare-token blocking needs no separate pass. Candidate-side vectors use

**sqrt** length normalization; document frequency is capped **per term kind** (20,000 for

tokens, 3,000 for n-grams).

**Country is a hard gate**, implemented by building a separate index per country label. An

unseen label gets its own index automatically — the test set's **France (259,452 S1 entities,

15.0%)** produced a third index with no code change and no filtering.

* **Candidate pairs generated (test):** 69,301,748 — exactly 40.00 per entity (min 28, max 40)
* **Reduction ratio vs the full cross product:** 1 − 69,301,748 / (1,732,544 × 9,969,589) = **0.999996**
* **Candidate recall (validation, 60,000 entities):** **0.961853**; 88.99% of entities have

  *all* their true matches retrieved

### How true matches were kept

Every blocking choice was measured against held-out candidate recall, and three plausible

ideas were **rejected because they made recall worse**:

| variant                          | cand recall @20/source | verdict      |
| -------------------------------- | ---------------------: | ------------ |
| tokens, L2 norm                  |                 0.9162 | baseline     |
| tokens, sqrt norm                |                 0.9318 | adopted      |
| + whole name/address keys        |                 0.8506 | **rejected** |
| + squashed-name key              |                 0.8386 | **rejected** |
| + char 4-grams @0.35             |                 0.9449 | adopted      |
| + address token weight 1.1       |                 0.9507 | adopted      |
| + df_cap 20,000 / gram cap 3,000 |                 0.9578 | adopted      |
| + **ALIAS** name tokens          |             **0.9619** | adopted      |

The two rejected variants add terms with document frequency ≈ 1, giving them IDF ≈ 15, which

dominates the candidate vector norm and *dilutes* every non-exact pair.

A dedicated audit (`src/audit_blocking.py`) classified all 8,793 validation misses

(**4.23%** of 208,091 true matches): 36.3% high-name-similarity-ranked-out, 28.7%

strong-address-ranked-out, 8.0% *identical* normalized name ranked out, 6.9%

transliteration, **3.9% unreachable, 0 country mismatches**. 96.1% were reachable but ranked

out at median rank 142, which is why raising *k* is inefficient: *k*=40 doubles the candidate

count to recover only 20.6% of misses. ALIAS was chosen instead because it recovers 19.1%

of misses **at zero extra candidates**.

---

## 4. Matching Model

### Preprocessing (`src/preprocessing.py`)

Raw fields are never mutated; every function adds a representation alongside them.

* **`romanize_indic`** — Indic→Latin transliteration whose tables are generated at import from

  `unicodedata.name()` (`DEVANAGARI LETTER KA` → `k`), covering ten Brahmi-derived blocks with

  one script-independent phoneme table and correct abugida handling (inherent *a*, virama,

  vowel signs, anusvara, visarga, nukta). Applied **before** ASCII folding, this takes the

  empty-`name_norm` rate from 26.9% to **0.0%**.

* **`normalize_name`** — NFKC → romanize → `&`→`and` → diacritic fold → lowercase →

  alphanumeric → single-letter-run collapse (`S.A.S` → `sas`).

* **`strip_legal_suffix`** → `name_core`, using a **75-token list discovered from the training

  ground truth** (`src/discover_tokens.py`): a token is noise when matched pairs disagree about

  its presence more than 70% of the time. This found the romanized suffixes a hand-written list

  would miss (`praiveta`, `limiteda`, `elaelapi`) plus DBA markers (`dba`, `formerly`, `www`).

  Words that merely *look* like suffixes but carry identity (`international`, `group`,

  `holdings`, `trust`) are deliberately excluded.

* **`normalize_address`** — same pipeline plus an abbreviation table unifying US, Indian and

  French street/unit words (`road`/`rd`, `rue`/`r`, `nagar`, `chowk`, `cedex`, …).

* **`normalize_country`** — lowercased token, **open set**: any unseen label becomes its own

  token; nothing is mapped to a fixed vocabulary or filtered.

* **Alias map** (`src/discover_aliases.py`) — learned from ground truth by measuring which

  A-only token substitutes for which B-only token: **108 address pairs** (179 tokens → 74

  canonical forms) and **126 name pairs**. It recovered US state codes (`tx`↔`texas`), Indian

  state codes (`maharashtra`↔`mh`, `karnataka`↔`ka`, `up`↔`uttar`/`pradesh`) and a

  romanized→English lexicon (`energy`↔`enarji`, `exports`↔`eksaportsa`, `food`↔`phuda`,

  `technology`↔`teknoloji`). Discovery is restricted to *training* entities.

### Features used (57 total, `src/features.py`)

* **Name string similarity (11):** exact `name_norm` / `name_core` / de-spaced equality;

  rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, `WRatio`;

  de-spaced `ratio` and Jaro-Winkler.

* **Name token sets (10):** Jaccard, containment, shared-token count, IDF-weighted overlap and

  IDF mass, first/last token agreement, acronym score, **alias-canonicalized Jaccard and

  containment**.

* **Address (19):** exact equality; `ratio` / `token_set_ratio` / `token_sort_ratio`; Jaccard,

  containment, shared count, IDF-weighted overlap and mass; numeric-token Jaccard,

  containment, one-sided counts and a conflict flag; postcode agreement and both-present;

  tail (city/state position) overlap; **alias-canonicalized Jaccard and containment**.

* **Shape (8):** name/address lengths and differences, token-count difference, address-missing flag.

* **Country / source / retrieval context (9):** country equality, country missing,

  source-is-S2 indicator, blocking cosine, blocking rank, candidate count, score margin and

  score ratio vs the entity's best candidate.

Country appears **only** as agreement + missingness flags — never one-hot — so an unseen

label cannot change the feature width or break inference.

Address similarity features are set to **NaN** (not 0) when either side has no address, because

a structural zero reads as genuine dissimilarity; `HistGradientBoostingClassifier` learns a

dedicated missing branch. 34.1% of the earlier model's false negatives were "missing address

on one side".

**Model type:** `sklearn.ensemble.HistGradientBoostingClassifier` (BSD-3-Clause), 400

iterations, learning rate 0.10, 63 leaves, 100 min samples/leaf, L2 = 1.0, seed 20260925.

Parameter count ~10⁵ — far under the 8-billion limit, and no external model weights are

downloaded. A gradient-boosted tree ensemble over engineered similarity features was chosen

over a neural/LLM matcher because the signal lives in the blocking and the features; 69.3M

pairs must be scored on 8 CPU cores with no GPU.

**Threshold selection method:** exhaustive sweep of 21 thresholds × 3 decision rules,

selected by **validation macro F₀.₅** — never accuracy, never left at 0.5.

### Training procedure

Trained on candidate pairs from a deterministic 250,000-entity sample of the *training*

Source-1 entities: **10,000,000 pairs, 832,924 positives (8.33%)**. Labels come from the

ground-truth edge list via an integer-key sorted join. Validation pairs are featurized

separately and never enter training. Model fit time 374 s; validation pair-level **AUC 0.99947,

average precision 0.99511**.

Top features by permutation importance (drop in average precision): `alias_addr_cont`

**0.0260**, `name_len_b` 0.0133, `name_partial` 0.0062, `name_idf_ovl` 0.0061, `name_tset`

0.0051. The learned alias containment is the single strongest feature.

---

## 5. Results & Error Analysis

### Validation methodology

The split is at the **Source-1-entity level**, never at the pair level, stratified by

(country, match-count bucket), seed 20260925: **2,074,412 training** and **132,409 validation**

entities, of which a fixed 60,000 are scored. Both sides carry an identical 5.585% singleton

rate. Verified this session: **zero entity overlap**, and the training and validation candidate

tables share no entity. Validation entities are blocked against the **full 10.3M** training

S2+S3 pool, so the haystack matches the real task.

The metric is re-implemented independently in `src/verify_model.py` directly from the problem

statement and does **not** import `src/evaluate.py`, so a bug in the metric module cannot

flatter the result. `src/evaluate.py` additionally self-tests against the worked example in the

problem statement (0.714286).

### Measured results

**F₀.₅ Score (macro, held-out validation — not a test score):**

|                        |       v1 |       v2 | **v3 (submitted)** |
| ---------------------- | -------: | -------: | -----------------: |
| **macro F₀.₅**         | 0.951895 | 0.954418 |       **0.956248** |
| macro precision        | 0.977404 | 0.976751 |       **0.979818** |
| macro recall           | 0.897231 | 0.907223 |       **0.905135** |
| candidate recall       | 0.957744 | 0.957744 |       **0.961853** |
| entities fully covered | 0.880906 | 0.880906 |       **0.889887** |
| singleton accuracy     | 0.955449 | 0.950331 |       **0.960867** |
| features               |       47 |       57 |                 57 |
| threshold              |     0.70 |     0.65 |               0.70 |

Each step was confirmed by a **paired** test over the same 60,000 entities (comparing two means

against their own standard errors overstates the uncertainty when both models are scored on

identical entities):

| step                                      | Δ macro F₀.₅ | paired SE |     t |       95% bootstrap CI |
| ----------------------------------------- | -----------: | --------: | ----: | ---------------------: |
| v1 → v2 (alias/containment/NaN features)  |    +0.002523 |  0.000281 | +8.99 | [+0.001982, +0.003080] |
| v2 → v3 (ALIAS blocking, τ fixed at 0.65) |    +0.001714 |  0.000249 | +6.89 | [+0.001235, +0.002212] |

Threshold: τ = 0.70 scores 0.956248 and τ = 0.65 scores 0.956132. That 0.000116 gap is

noise-level. τ = 0.70 was retained because it also had higher validation precision

(0.9798 vs 0.9778) and singleton accuracy (0.9609 vs 0.9506).

### Singleton handling

5.58% of training entities have no match. The decision layer can always return the empty set —

no top-1 match is ever forced. At τ = 0.70 the model correctly predicts empty for

**3,192 of 3,322** validation singletons (**96.09%**). Two other rules were implemented and

measured but lost: a best-match margin rule, and a per-entity expected-F₀.₅ maximiser

(accept the *m* that maximises a plug-in E[F₀.₅], comparing against ∏(1−p) for the empty set) —

the latter reached only 0.95364 because it declares too few singletons.

### Common false positives (wrong merges)

At the v2 operating point: 2,476 false positives against 185,764 true positives, of which

**161 land on singleton entities** (the most expensive error class — a false merge there costs

that entity its full 1.0). Patterns: 41.0% moderate-on-both-fields, 24.6% missing address on

one side, 17.2% "both look identical", 13.8% same address / different name. The genuinely hard

residual is near-identical records at the same address that ground truth says are distinct —

e.g. `eastern laboratories care` vs `eastern laboratories care corp` at the same address, and

an identical name at `98 decatur st` vs `119 decatur st` where only the house number differs.

This is what motivated the numeric-containment and numeric-conflict features.

### Common false negatives (missed matches)

13,534 retrieved-but-rejected plus 8,793 never-retrieved. **The model rejecting retrieved true

matches is 1.5× larger than the blocking loss**, so it is the bigger remaining lever. 34.1% of

rejections have an empty address on one side (addressed by NaN-encoding), 39.6% are moderate on

both fields. Blocking misses are 96.1% ranking failures rather than coverage failures.

---

## 6. Conclusion

A carefully measured classical pipeline — country-gated IDF retrieval, 57 engineered

similarity features, and a gradient-boosted classifier with an F₀.₅-tuned per-entity threshold

— reaches **0.956248 held-out validation macro F₀.₅** with 0.9798 precision while still

correctly identifying 96.1% of singletons. The decisive lessons were that the highest-leverage

work was *data forensics*, not modelling: one line of naive ASCII folding would have silently

discarded 1.35M business names, and a default CSV read would have corrupted others. Three

plausible blocking ideas were rejected on measurement, and the two biggest wins — Unicode-derived

romanization and a ground-truth-learned alias map — both came from looking at what the data

actually contained.

---

## Appendix

### A. Code Artefacts

Complete runnable pipeline in `code/business_entity_resolution/` (all sources under `src/`,

plus `README.md` and a pinned `requirements.txt`). `run_all.sh` reproduces everything from the

raw TSVs to the submission zip. Entry points, in order:

| module                                                                | role                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| `src/config.py`                                                       | paths, seed 20260925, tunables                         |
| `src/io_utils.py`                                                     | TSV I/O — tab-separated, **quoting disabled**          |
| `src/preprocessing.py`                                                | Unicode/Indic normalization, legal suffixes, addresses |
| `src/profile_data.py`                                                 | data profiling                                         |
| `src/prepare.py`                                                      | normalize + cache all sources as parquet               |
| `src/discover_tokens.py`                                              | derives the 75-token noise list from ground truth      |
| `src/discover_aliases.py`                                             | derives the alias map from ground truth                |
| `src/make_split.py`                                                   | Source-1-level stratified validation split             |
| `src/evaluate.py`                                                     | competition macro F₀.₅ (self-tested)                   |
| `src/blocking.py`, `src/gen_candidates.py`                            | candidate generation                                   |
| `src/features.py`, `src/build_features.py`                            | 57 pairwise features, streamed                         |
| `src/train.py`, `src/decide.py`                                       | model training, decision rules                         |
| `src/verify_model.py`, `src/compare_models.py`                        | independent re-measurement, paired tests               |
| `src/audit_blocking.py`, `src/probe_keys.py`, `src/error_analysis.py` | diagnostics                                            |
| `src/preflight.py`, `src/infer.py`, `src/audit_output.py`             | final inference and checks                             |

Reproduce the two output files:

```bash
python3 -m src.prepare && python3 -m src.discover_tokens && python3 -m src.make_split

python3 -m src.discover_aliases && python3 -m src.features train && python3 -m src.features test

python3 -m src.gen_candidates --split test --queries all --topk 20 \
    --w-alias-name-tok 0.9 --tag all_k20_alias

python3 -m src.infer --model models/matcher_v3.joblib \
    --cand artifacts/cache/cand_test_all_k20_alias.parquet
```

### B. Additional Results

**Final test inference** (no ground truth available locally, so these are output statistics,

not a score): 1,732,544 Source-1 entities, 69,301,748 candidate pairs scored,

**5,529,585 matches** emitted (3.1916 per entity), **107,055 empty predictions (6.18%)**,

0 entities without candidates. Scoring 1,814 s, total 1,872 s, peak RSS 4.71 GB.

Reassuringly, the test output distribution closely tracks validation: 3.19 vs 3.16 matches per

entity and 6.18% vs 6.18% empty predictions — consistent behaviour on a test set that is 15%

France, a country absent from training.

**Computational considerations** (8 CPU cores, 16 GB RAM, 4 GB swap, no GPU; every stage run

one at a time):

| stage                                   | runtime | peak RSS |
| --------------------------------------- | ------: | -------: |
| profile raw data                        |   194 s |   3.2 GB |
| normalize + cache 7 files               |   122 s |   4.0 GB |
| test candidate generation (69.3M pairs) | 2,696 s |  6.08 GB |
| model training (10M × 57)               |   722 s |  6.88 GB |
| test inference (69.3M pairs)            | 1,872 s |  4.71 GB |
| output audit                            |   115 s |  0.64 GB |

Memory techniques that were necessary, not decorative: entity ids stored as `int32` (ids are

pure digits ≤ 9 chars); all features `float32`; features streamed in 1–3M-pair chunks (the full

69.3M × 57 matrix would be 15.8 GB); the training matrix **preallocated** rather than

`np.concatenate`-d from a chunk list (which held two 2.3 GB copies at once); the blocking index

built in two passes that re-tokenize instead of caching per-document column lists (worth 1.4 GB

on a 5M-record source); `eid`→row lookup by sorted-array `searchsorted` instead of a 5M-entry

dict (~40 MB vs ~400 MB); and the output audit streams both TSVs in lockstep rather than

loading `candidate_pairs.tsv` into a dict (~6 GB).

### C. Validation Hygiene — full disclosure

`src/discover_tokens.py` derives its 75-token noise list from ground-truth pairs sampled across

**all** Source-1 entities, including validation ones, and that list feeds `name_core` → blocking,

features and IDF. Using all training ground truth is legitimate for test inference — it is

provided training data, not external data — but it means the validation figures above are **not

mathematically perfectly hermetic**. Measured effect: restricting discovery to training entities

changes the list by **exactly one token** (`kanstrakshansa`), and that token appears in the

*train-only* list, so the difference is sampling noise from a smaller edge pool rather than

validation information leaking in. `src/discover_aliases.py` **is** restricted to training

entities. We report this rather than claim a clean-room guarantee we did not establish.

### D. Fair-Play Compliance

* The only inputs are the seven provided TSV files. No network access occurs at any point;

  `src/preflight.py` asserts that no networking or geocoding module is imported anywhere in the

  pipeline.

* **No** external database, business registry, commercial entity-resolution API, geocoding

  service, gazetteer, external dataset, LLM, or downloaded model weights are used.

* The Indic→Latin romanization table is generated from Python's bundled Unicode character

  database (`unicodedata.name`) — a character-encoding property, in the same class as the

  `Rd`↔`Road` abbreviation table the starter code itself ships. It maps code points to letters;

  it does not map any name to a real-world entity.

* The legal-suffix list and the alias map are **derived from the provided training ground truth**

  by `src/discover_tokens.py` and `src/discover_aliases.py`, and both derivations ship in the

  submission so they can be re-run and audited.

* The matcher is trained from scratch on the provided training data with

  `HistGradientBoostingClassifier` (BSD-3-Clause), ~10⁵ parameters, satisfying the

  MIT/Apache-2.0-compatible and ≤ 8B-parameter requirements.

* No test label is used anywhere (none exists), and no test record was inspected to hand-craft a

  rule or a match.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
