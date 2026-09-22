# MBES Habitat Framework — MER 2026 Reproducibility Snapshot

This repository contains a publication-specific snapshot of the analysis code supporting the results reported in the associated *Marine Environmental Research* manuscript.

The repository documents the computational workflow used to evaluate multi-frequency multibeam echosounder (MBES)-based prediction of marine vegetated habitats at two shallow coastal sites, **Hujeong** and **Bongpyeong**, Republic of Korea.

This release is intended to support transparency, inspection, and reproducibility of the reported analyses. It is **not** presented as a complete raw-data-to-publication archive because the underlying raw MBES survey data are not redistributed here.

## Study overview

The analysis uses MBES observations acquired at three frequencies:

- 200 kHz
- 300 kHz
- 400 kHz

The publication workflow evaluates:

- seven canonical feature sets;
- four machine-learning classifiers;
- within-site validation;
- reciprocal cross-site validation; and
- spatial block cross-validation.

The repository contains the project configurations, experiment definitions, analysis pipeline, testing framework, and supporting documentation associated with the publication-specific workflow.

## Repository contents

```text
.
├── 01_data/
├── 02_projects/
├── 03_experiments/
├── 04_pipeline/
├── docs/
├── tests/
├── .gitignore
├── CITATION.cff
├── FILE_MANIFEST.csv
├── LICENSE_STATUS.md
├── README.md
├── RELEASE_NOTES.md
├── REQUIREMENTS.md
├── environment.yml
└── requirements.txt
```

### `01_data/`

Input-data structure documentation only. **Raw survey data are not included.**

The workflow expects site-specific input data to be supplied locally in the documented directory structure, including:

```text
01_data/HJ_202204/
01_data/BP_202507/
```

Users wishing to execute the workflow must provide the required authorized MBES inputs in these locations.

### `02_projects/`

Portable MER 2026 project and site configuration files.

### `03_experiments/`

Configuration files defining the feature sets, model experiments, validation analyses, candidate selection, and publication-oriented settings.

### `04_pipeline/`

The staged analysis workflow used for the publication, including **Stages 00–13 and Stage 99**.

### `docs/`

Supporting workflow and reproducibility documentation.

### `tests/`

Automated tests and quality-assurance checks associated with the publication-specific framework.

### `FILE_MANIFEST.csv`

A file-level manifest of the public code package, retained for package integrity and provenance.

### `RELEASE_NOTES.md`

Scientific provenance and release-history information, including the distinction between the frozen scientific run and subsequent non-scientific maintenance.

### `REQUIREMENTS.md`

Software compatibility and environment information for the public release.

### `LICENSE_STATUS.md`

The current software-rights and licensing status.

## Workflow

The repository is organized as a staged analysis pipeline. The authoritative workflow order is defined by the scripts and configuration files under `04_pipeline/`, `02_projects/`, and `03_experiments/`.

The workflow covers the major analysis steps required for:

1. project and input preparation;
2. data alignment and harmonization;
3. feature generation;
4. reference-label preparation and sampling;
5. model training;
6. prediction;
7. validation;
8. candidate evaluation and selection;
9. mapping and export;
10. publication-oriented source-data generation; and
11. final quality-assurance checks.

Users should follow the pipeline stage order and the configuration files supplied with this release rather than treating individual scripts as independent analyses.

## Scientific provenance

| Item | Value |
|---|---|
| Frozen scientific run | `run_20260916_035737_2df29152` |
| Source Git commit | `cc4fc5f6b1c66bdd60633589c235230acfec6165` |
| Run completed | 16 September 2026 |

Subsequent framework changes were limited to non-scientific maintenance associated with publication preparation, including common-valid exports, map previews, north arrows and scale bars, publication terminology, output organization, and output-contract checks.

These subsequent changes did **not** alter the validation predictions, final probability rasters, or candidate average-precision values underlying the reported scientific results.

See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for details.

## Reproducibility scope

This repository provides a **publication-specific reproducibility snapshot** of the analysis code and associated configuration.

It includes the software logic, experiment definitions, project configuration, tests, and environment documentation needed to inspect the reported analytical workflow and to execute it when appropriate authorized inputs are supplied.

Raw MBES survey data are not included. Accordingly, this repository should not be interpreted as a fully self-contained archive enabling unrestricted raw-data-to-publication reproduction.

A companion **Mendeley Data deposit** containing the derived publication source data associated with this study has been submitted under the reserved DOI [10.17632/7dz3gn5zmg.1](https://doi.org/10.17632/7dz3gn5zmg.1). The publication and access status of the dataset is governed by the corresponding Mendeley Data record.

## Mapping threshold

Binary prediction masks associated with the publication workflow use the classification threshold:

```text
P ≥ 0.5
```

where `P` denotes the predicted probability of the vegetated class.

## Data availability

The GitHub code release intentionally excludes:

- raw MBES survey data;
- model binaries;
- proprietary or commercial imagery;
- manuscript artwork;
- internal execution records;
- historical runs;
- caches and temporary files;
- large derived raster products; and
- third-party assets for which redistribution has not been authorized.

Raw MBES data are retained as part of ongoing institutional research activities and are not redistributed through this repository.

Access to underlying survey data may be considered on reasonable request, subject to applicable institutional, project, and data-management constraints.

## Companion Mendeley Data deposit

A companion Mendeley Data deposit contains compact derived publication data supporting the reported results.

**Reserved DOI:** [10.17632/7dz3gn5zmg.1](https://doi.org/10.17632/7dz3gn5zmg.1)

The compact data package includes derived source data supporting the reported publication results, including:

- panel-level source data for manuscript Figures 2–10;
- panel-level source data for Supplementary Figures S1–S9;
- canonical source data for manuscript Tables 1–4;
- canonical source data for Supplementary Tables S1–S10;
- aggregate validation summaries;
- candidate-selection datasets;
- mapping summaries;
- publication metadata;
- figure/table mappings; and
- a column-level data dictionary.

The Figure 1 locator is not included in the source-data deposit because it was manually prepared using external basemap imagery.

Large optional raster products are not part of the compact data deposit. These include the full-resolution probability rasters, prediction masks, and common-valid GeoTIFF products.

The derived publication data and metadata in the Mendeley Data deposit are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## Relationship between the code and data releases

The GitHub repository and Mendeley Data deposit serve complementary purposes:

- The **GitHub repository** provides the publication-specific code, configurations, tests, and workflow documentation.
- The **Mendeley Data deposit** provides derived publication source data used to verify the reported figures, tables, validation summaries, candidate-selection results, and mapping summaries.

Together, these resources are intended to improve transparency and reproducibility without redistributing raw institutional MBES survey data or restricted third-party assets.

## Software environment

Environment information is provided in:

- [`REQUIREMENTS.md`](REQUIREMENTS.md)
- [`environment.yml`](environment.yml)
- [`requirements.txt`](requirements.txt)

The compatibility environment documented for this release was captured using **Python 3.11.14** on **macOS arm64**.

This environment information is provided as a practical compatibility reference and should not be interpreted as an exact reconstruction of every historical software build used during development.

A Conda-based environment is recommended because the workflow depends on multiple geospatial Python libraries and their system-level dependencies. See [`REQUIREMENTS.md`](REQUIREMENTS.md) for additional details.

## Citation

| Resource | Identifier |
|---|---|
| GitHub repository | <https://github.com/neobassist/MBES-Habitat-Framework> |
| GitHub release/tag | v1.0.0 |
| Companion Mendeley Data DOI | [10.17632/7dz3gn5zmg.1](https://doi.org/10.17632/7dz3gn5zmg.1) |

Please also consult [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

When using these resources, please cite the associated manuscript and the relevant software and data records as appropriate.

## License and reuse

The authoritative framework source did not contain an explicit public software license when this publication package was assembled. No software license has therefore been assigned implicitly or retroactively to the code in this repository.

See [`LICENSE_STATUS.md`](LICENSE_STATUS.md) for the current software-rights status. A software license may be added following confirmation by the relevant authors and rights holder(s).

> **Note:** The CC BY 4.0 license associated with the companion Mendeley Data deposit applies to the deposited derived data and metadata, **not** to the software in this GitHub repository.

Until explicit software reuse terms are assigned, public availability of this repository should not be interpreted as granting unrestricted software reuse beyond applicable law and the terms explicitly stated in the repository.

## Contact

For questions concerning this publication-specific code snapshot or its relationship to the companion data deposit, please contact:

**SoonYoung Choi**<br>
Korea Institute of Ocean Science and Technology (KIOST)<br>
sychoi@kiost.ac.kr