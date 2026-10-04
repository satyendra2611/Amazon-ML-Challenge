#!/usr/bin/env bash
# Reproduce the whole pipeline end to end, from the raw TSVs to the submission zip.
#
# Usage: ./run_all.sh [team_name]
#
# Runs exactly ONE heavy stage at a time.  The heaviest stage peaks at ~6.9 GB RSS;
# on a 16 GB machine two concurrent stages will thrash swap, so do not parallelise.
set -euo pipefail

TEAM="${1:-team}"
K=20
ALIAS=0.9          # ALIAS blocking weight (v3).  0.0 reproduces the v2 baseline.
cd "$(dirname "$0")"
mkdir -p logs

step () { echo; echo "=============== $* ==============="; }

step "0/14 metric self-test (reproduces the worked example in the problem statement)"
python3 -m src.evaluate

step "1/14 profile raw data"
python3 -m src.profile_data              | tee logs/01_profile.log

step "2/14 normalize + cache all sources as parquet"
python3 -m src.prepare                   | tee logs/02_prepare.log

step "3/14 discover legal-suffix / noise tokens from the training ground truth"
python3 -m src.discover_tokens           | tee logs/03_tokens.log

step "4/14 deterministic Source-1-level validation split"
python3 -m src.make_split                | tee logs/04_split.log

step "5/14 corpus IDF tables"
python3 -m src.features train            | tee logs/05_idf.log
python3 -m src.features test             | tee -a logs/05_idf.log

step "6/14 learn the token-substitution alias map from the ground truth"
python3 -m src.discover_aliases          | tee logs/06_aliases.log

step "7/14 candidate generation with ALIAS (training sample + validation)"
python3 -m src.gen_candidates --split train --queries train --sample 250000 \
    --topk $K --w-alias-name-tok $ALIAS --tag trainsub_k${K}_alias \
    | tee logs/07_cand_train.log
python3 -m src.gen_candidates --split train --queries val \
    --topk $K --w-alias-name-tok $ALIAS --tag val_k${K}_alias \
    | tee -a logs/07_cand_train.log

step "8/14 train the matcher, choose the decision rule by validation macro F0.5"
python3 -m src.train \
    --train-cand artifacts/cache/cand_train_trainsub_k${K}_alias.parquet \
    --val-cand   artifacts/cache/cand_train_val_k${K}_alias.parquet \
    --exp v3 --chunk 1000000 --max-iter 400 --lr 0.10 \
    --max-leaf 63 --min-leaf 100 --l2 1.0 | tee logs/08_train.log

step "9/14 independently re-measure the saved model"
python3 -m src.verify_model --model models/matcher_v3.joblib \
    --cand artifacts/cache/cand_train_val_k${K}_alias.parquet --sweep \
    | tee logs/09_verify.log

step "10/14 error analysis + blocking failure audit"
python3 -m src.error_analysis --model models/matcher_v3.joblib \
    --cand artifacts/cache/cand_train_val_k${K}_alias.parquet | tee logs/10_errors.log
python3 -m src.audit_blocking            | tee -a logs/10_errors.log

step "11/14 final pre-flight"
python3 -m src.preflight                 | tee logs/11_preflight.log

step "12/14 candidate generation (TEST) -- this IS the set the matcher scores"
python3 -m src.gen_candidates --split test --queries all \
    --topk $K --w-alias-name-tok $ALIAS --tag all_k${K}_alias \
    | tee logs/12_cand_test.log

step "13/14 test inference -> output/*.tsv, then audit + official validator"
python3 -m src.infer --model models/matcher_v3.joblib \
    --cand artifacts/cache/cand_test_all_k${K}_alias.parquet \
    --chunk 2000000                      | tee logs/13_infer.log
python3 -m src.audit_output              | tee logs/13_audit_output.log
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test              | tee logs/13_validate.log

step "14/14 package submission"
python3 -m src.package_submission --team-name "$TEAM" | tee logs/14_package.log

echo; echo "DONE -> ${TEAM}_submission.zip"
