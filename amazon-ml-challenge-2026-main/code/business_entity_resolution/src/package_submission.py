"""Build the final <team_name>_submission.zip in the exact required structure.

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   `-- candidate_pairs.tsv
    |-- code/
    |   `-- business_entity_resolution/
    |       |-- src/
    |       |-- README.md
    |       `-- requirements.txt
    `-- Documentation_template.md

Runs the official validator first and refuses to package unless it passes.

Run:  python3 -m src.package_submission --team-name my_team
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import config as C

CODE_PREFIX = "code/business_entity_resolution"


def run_validator(matching: Path, candidate: Path, test_dir: Path,
                  check_ids: bool) -> tuple[bool, str]:
    cmd = [sys.executable, str(C.ROOT / "utils" / "validate_submission.py"),
           "--matching", str(matching), "--candidate", str(candidate),
           "--test-dir", str(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.ROOT))
    return p.returncode == 0, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--skip-validator", action="store_true")
    ap.add_argument("--check-ids", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    matching = C.OUTPUT_DIR / "matching_results.tsv"
    candidate = C.OUTPUT_DIR / "candidate_pairs.tsv"
    for f in (matching, candidate):
        if not f.exists():
            print(f"ERROR: {f} missing -- run `python3 -m src.infer` first")
            return 1

    if not a.skip_validator:
        print("[package] running the official validator ...")
        ok, out = run_validator(matching, candidate, C.TEST_DIR, a.check_ids)
        print(out)
        if not ok:
            print("ERROR: validator did not pass -- fix the pipeline, do not "
                  "hand-edit the outputs.")
            return 1

    zip_path = Path(a.out) if a.out else C.ROOT / f"{a.team_name}_submission.zip"
    doc = C.ROOT / "Documentation_template.md"
    readme = C.ROOT / "SUBMISSION_README.md"
    reqs = C.ROOT / "requirements.txt"
    for f in (doc, readme, reqs):
        if not f.exists():
            print(f"ERROR: {f} missing")
            return 1

    src_files = sorted(p for p in (C.ROOT / "src").glob("*.py"))
    if not src_files:
        print("ERROR: no source files found under src/")
        return 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(matching, "output/matching_results.tsv")
        z.write(candidate, "output/candidate_pairs.tsv")
        for p in src_files:
            z.write(p, f"{CODE_PREFIX}/src/{p.name}")
        z.write(readme, f"{CODE_PREFIX}/README.md")
        z.write(reqs, f"{CODE_PREFIX}/requirements.txt")
        z.write(C.ROOT / "run_all.sh", f"{CODE_PREFIX}/run_all.sh")
        # The trained matcher is 2.9 MB and lets a reviewer reproduce the outputs
        # without a 12-minute training run.  The pipeline can regenerate it from the
        # provided data alone, so this is a convenience, not a dependency.
        for rel in ("models/matcher_v3.joblib", "models/PRODUCTION.md"):
            pm = C.ROOT / rel
            if pm.exists():
                z.write(pm, f"{CODE_PREFIX}/{rel}")
        # the pipeline's own validator copy, so the package is self-contained
        z.write(C.ROOT / "utils" / "validate_submission.py",
                f"{CODE_PREFIX}/utils/validate_submission.py")
        z.write(doc, "Documentation_template.md")
        # measured artifacts that back the numbers in the write-up
        for rel in ("artifacts/profile/profile.txt",
                    "artifacts/profile/match_anatomy.txt",
                    "artifacts/profile/legal_tokens.json",
                    "artifacts/profile/aliases.json",
                    "artifacts/experiments.csv",
                    "artifacts/validation/split_summary.json",
                    "artifacts/validation/verify_models_matcher_v3.json",
                    "artifacts/validation/compare_v2_v3_tau065.json",
                    "artifacts/validation/audit_blocking.json",
                    "artifacts/validation/audit_output.json",
                    "artifacts/errors/summary.json"):
            p = C.ROOT / rel
            if p.exists():
                z.write(p, f"{CODE_PREFIX}/{rel}")

    names = zipfile.ZipFile(zip_path).namelist()
    print(f"\n[package] {zip_path.name}  "
          f"{zip_path.stat().st_size/1e6:.1f} MB  {len(names)} entries")
    for n in sorted(names):
        print(f"    {n}")

    required = {"output/matching_results.tsv", "output/candidate_pairs.tsv",
                f"{CODE_PREFIX}/README.md", f"{CODE_PREFIX}/requirements.txt",
                "Documentation_template.md"}
    missing = required - set(names)
    if missing:
        print(f"ERROR: zip is missing {sorted(missing)}")
        return 1
    if not any(n.startswith(f"{CODE_PREFIX}/src/") for n in names):
        print("ERROR: zip has no code/business_entity_resolution/src/ files")
        return 1
    print("\n[package] structure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
