#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".ps1", ".md", ".txt", ".csv", ".yml", ".yaml", ".json",
    ".cff", ".gitignore", ".gitattributes",
}
RAW_OR_PRIVATE_SUFFIXES = {
    ".dcm", ".dicom", ".nii", ".gz", ".mha", ".mhd", ".nrrd", ".xml",
    ".pt", ".pth", ".ckpt", ".onnx", ".pem", ".key",
}
PATH_PATTERN = re.compile(
    r"(?i)(?:^|[\s\"'(])(?:[A-Z]:\\(?:Users|data|code|ProgramData|"
    r"Projects|Documents|Downloads)\\)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)"
    r"\s*[:=]\s*[\"'][^\"']+[\"']"
)

def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    errors: list[str] = []

    required = [
        "README.md", "LICENSE", "CITATION.cff", ".zenodo.json", ".gitignore",
        "folds/Protocol_V3_Patient_Fold_Assignment.csv",
        "folds/Protocol_V3_Sanitised_ROI_Manifest.csv",
        "oof_predictions/Reproduced_Baseline_OOF_Predictions.csv",
        "oof_predictions/Ordinal_Model_OOF_Predictions.csv",
        "oof_predictions/Ordinal_Auxiliary_Risk_OOF_Predictions.csv",
    ]
    for rel in required:
        if not (root / rel).exists():
            errors.append(f"Missing required file: {rel}")

    # Metadata.
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    for value in [
        "https://github.com/pengwuke/lss-foraminal-ordinal-grading",
        "https://orcid.org/0009-0003-5791-3783",
    ]:
        if value not in citation:
            errors.append(f"CITATION.cff missing: {value}")
    if "USERNAME" in citation:
        errors.append("CITATION.cff still contains USERNAME placeholder.")

    try:
        zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
        creator = zenodo["creators"][0]
        if creator.get("orcid") != "0009-0003-5791-3783":
            errors.append(".zenodo.json ORCID mismatch.")
    except Exception as exc:
        errors.append(f"Invalid .zenodo.json: {exc}")

    # File content scan.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".git", "outputs", "__pycache__", ".release_backup"} for part in relative.parts):
            continue
        lower_name = path.name.lower()
        suffixes = "".join(path.suffixes).lower()
        if path.suffix.lower() in RAW_OR_PRIVATE_SUFFIXES or suffixes in {".nii.gz"}:
            errors.append(f"Raw/private binary extension found: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore", ".zenodo.json", "CITATION.cff"
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PATH_PATTERN.search(line):
                errors.append(
                    f"Personal absolute path: {relative}:{line_number}: {line.strip()}"
                )
            if SECRET_PATTERN.search(line):
                errors.append(
                    f"Possible embedded secret: {relative}:{line_number}: {line.strip()}"
                )

    # Python syntax without producing __pycache__.
    for path in (root / "scripts").glob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Python syntax error: {path.name}: {exc}")

    # Frozen counts.
    count_checks = {
        "folds/Protocol_V3_Patient_Fold_Assignment.csv": 468,
        "folds/Protocol_V3_Sanitised_ROI_Manifest.csv": 2978,
        "oof_predictions/Reproduced_Baseline_OOF_Predictions.csv": 2978,
        "oof_predictions/Ordinal_Model_OOF_Predictions.csv": 2978,
        "oof_predictions/Ordinal_Auxiliary_Risk_OOF_Predictions.csv": 2978,
    }
    for rel, expected in count_checks.items():
        path = root / rel
        if path.exists():
            actual = csv_count(path)
            if actual != expected:
                errors.append(f"Row count mismatch: {rel}: {actual} != {expected}")

    if errors:
        print("=" * 96)
        print("PUBLIC RELEASE CHECK FAILED")
        print("=" * 96)
        for error in errors:
            print(f"- {error}")
        return 1

    print("=" * 96)
    print("PUBLIC RELEASE CHECK PASSED")
    print("GitHub: https://github.com/pengwuke/lss-foraminal-ordinal-grading")
    print("ORCID: 0009-0003-5791-3783")
    print("Patients/ROIs: 468/2978")
    print("=" * 96)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
