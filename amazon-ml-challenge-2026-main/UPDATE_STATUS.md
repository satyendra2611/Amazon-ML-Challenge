# Updated package status

This package preserves the original `output/matching_results.tsv` and
`output/candidate_pairs.tsv` and the original `matcher_v3.joblib` model. The
leaderboard score reported by the team for that output was 0.941. The output
files in this archive have **not** been shown to score higher on the leaderboard.

The code update adds `src/experiment_k40.py`, a controlled top-40-per-source
blocking and retraining experiment. See the new "Measured improvement
experiment" section of `code/business_entity_resolution/README.md`.

Using the supplied dataset, top-40 candidate recall rose from 0.961853 to
0.970484, but its retrained model scored 0.955419 held-out macro F0.5, below
the submitted model's 0.956248. The original matcher on top-40 candidates
scored 0.953358. The experiment was therefore rejected for the final output.
The final TSVs passed the official validator with `--check-ids` against the
supplied test files: 1,732,544 Source-1 rows, all IDs valid.

To reproduce the experiment, place the seven original competition TSVs under
`code/business_entity_resolution/dataset/{train,test}/`, or set
`BER_DATA_DIR` to their directory. Run preparation steps 1-5b in the README,
then `python -m src.experiment_k40` from the project directory. Only if the
independently verified validation gain is sufficient should you run
`python -m src.experiment_k40 --produce-test` to replace both output TSVs.

No claim of a 0.99 score is made. The challenge test labels are unavailable.
