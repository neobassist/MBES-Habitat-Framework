# Software requirements

The recommended environment is the included `environment.yml`, which records Python 3.11.14 and a mutually importable package set captured on macOS arm64 on 12 August 2026. It is a compatibility environment, not a claim that every archived artifact was created with exactly those package builds.

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate mbes-habitat-framework
```

The core runtime uses NumPy, pandas, SciPy, scikit-learn, Rasterio, GeoPandas, Shapely, PyProj, Fiona, GDAL, Matplotlib, Joblib, PyYAML, LightGBM, XGBoost, and CatBoost. Native geospatial libraries are easiest to install consistently from `conda-forge`. `requirements.txt` records the same Python-level versions for environments where pip installation is appropriate, but GDAL and related native libraries may require platform-specific system packages.

The complete model-training and mapping workflow is computationally intensive and requires the compatible source rasters described in `01_data/README.md`. Package QA is limited to syntax checks, configuration parsing, imports exercised by the included regression test, and consistency checks that do not require restricted raw data.
