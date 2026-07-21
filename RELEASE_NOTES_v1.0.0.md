# Protocol V3 reproducibility package v1.0.0

First public reproducibility release for the patient-level ordinal grading
study of lumbar foraminal stenosis.

## Included

- fixed 468-patient five-fold assignment;
- sanitised manifest for 2,978 expert ROIs;
- sanitised OOF predictions for the three primary models;
- locked grading, calibration, bootstrap, and error-migration source tables;
- Protocol V3 training, aggregation, bootstrap, and diagnostic scripts;
- repository-relative PowerShell pipeline;
- SHA256 checksums and release-readiness scanner.

## Frozen primary design

- main seed 42;
- 60% DeiT + 40% CNN probability fusion;
- CORAL risk-loss weight 0.50;
- 5,000 patient-cluster bootstrap replicates;
- cross-fitted thresholds.

## Excluded

Raw MRI data, DICOM files, source annotation XML/PNG files, trained model
checkpoints, local absolute paths, credentials, and identifiable information
are not included.

## Source dataset

- https://doi.org/10.17632/rgb77xm3jf.4
- https://doi.org/10.1038/s41597-026-07138-x
