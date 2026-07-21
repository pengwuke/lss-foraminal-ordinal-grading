# Patient-level ordinal grading of lumbar foraminal stenosis

Reproducibility repository for the Protocol V3 Full-468 study associated with:

> Patient-level ordinal grading of lumbar foraminal stenosis on sagittal MRI:
> reproduction of a dual-branch baseline and safety-oriented evaluation using
> five-fold cross-validation

**Author:** Wuke Peng  
**Affiliation:** School of Smart Health, Chongqing Polytechnic University of Electronic Technology  
**Email:** 202628001@cquet.edu.cn  
**ORCID:** https://orcid.org/0009-0003-5791-3783  
**Repository:** https://github.com/pengwuke/lss-foraminal-ordinal-grading

## Public source dataset

Raw MRI data are not redistributed in this repository.

- Dataset: https://doi.org/10.17632/rgb77xm3jf.4
- Dataset paper: https://doi.org/10.1038/s41597-026-07138-x

## Frozen analysis design

- 468 evaluable patients
- 2,978 expert-defined foraminal ROIs
- Patient-disjoint five-fold cross-validation
- Main training seed: 42
- Primary fusion: 60% DeiT + 40% CNN branch
- Patient-cluster bootstrap: 5,000 replicates
- Cross-fitted review thresholds selected from the other four folds

## Repository structure

```text
.
├── CITATION.cff
├── LICENSE
├── README.md
├── .zenodo.json
├── .gitignore
├── environment.yml
├── scripts/
├── folds/
├── oof_predictions/
├── source_data/
├── docs/
├── tools/
└── checksums/
```

## Reproduction

The training scripts require a local ROI manifest that points to a separately
downloaded copy of the public dataset. Personal machine paths are not stored in
this repository.

The primary pipeline uses repository-relative script and output directories:

```powershell
& ".\scripts\59_run_final_main_pipeline.ps1" `
  -InputManifest "<LOCAL_ROI_MANIFEST.csv>" `
  -MainSeed 42 `
  -BootstrapReps 5000 `
  -ContinueOnError
```

When the fixed local Protocol V3 manifest already exists under
`outputs\protocol_v3_full468`, use:

```powershell
& ".\scripts\59_run_final_main_pipeline.ps1" `
  -SkipFoldBuild `
  -MainSeed 42 `
  -BootstrapReps 5000 `
  -ContinueOnError
```

See `docs/PORTABLE_USAGE.md` and `docs/RUNBOOK.md`.

## Data governance

- No raw MRI, DICOM, XML, or source PNG files are included.
- No names, dates of birth, accession numbers, credentials, or local absolute
  paths are included.
- Public dataset folder identifiers are retained only for reproducible
  patient-level fold assignment.

## Citation

GitHub reads `CITATION.cff`. Zenodo reads `.zenodo.json` when archiving GitHub
releases. After release `v1.0.0` is archived, add the version DOI badge and DOI
to this README.

## Licence

The repository uses the MIT License for original repository code and
documentation. Before public release, confirm that any code adapted from an
upstream project retains all required upstream licence and attribution notices.
The source MRI dataset remains governed by its original repository terms.
