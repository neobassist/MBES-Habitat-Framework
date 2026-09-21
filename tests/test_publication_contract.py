#!/usr/bin/env python3
"""Focused regression checks for the publication/output synchronization contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "04_pipeline"
sys.path.insert(0, str(PIPELINE))

from utils.config_io import RuntimePathBinding, ensure_output_dirs  # noqa: E402
from utils.plotting import (  # noqa: E402
    PREFERRED_SANS_FONTS,
    configure_publication_font,
    nice_scale_length,
)
from utils.raster_io import write_raster  # noqa: E402


def _load_numeric_stage(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PublicationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage01 = _load_numeric_stage("01_align.py", "stage01_contract")
        cls.stage13 = _load_numeric_stage("13_report_tables_figures.py", "stage13_contract")
        cls.publication = yaml.safe_load(
            (ROOT / "03_experiments/MER_2026_DB/publication.yaml").read_text(encoding="utf-8")
        )

    def test_stage_owned_directories_exclude_obsolete_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            binding = RuntimePathBinding(
                config_root=ROOT,
                data_root=base / "data",
                data_locator="01_data",
                framework_code_root=ROOT,
                output_root=base / "outputs",
                docs_root=base / "docs",
            )
            config = {
                "paths": {
                    "framework_root": ".",
                    "data_root": "01_data",
                    "project_root": "02_projects/MER_2026_DB",
                    "experiment_root": "03_experiments/MER_2026_DB",
                    "pipeline_root": "04_pipeline",
                    "output_root": "05_outputs/MER_2026_DB",
                    "docs_root": "06_docs/projects/MER_2026_DB",
                }
            }
            dirs = ensure_output_dirs(config, stage_name="13", runtime_binding=binding)
            self.assertTrue(dirs["report_outputs"].is_dir())
            for obsolete in ("export_asc", "figures", "tables", "habitat_maps"):
                self.assertFalse((binding.output_root / obsolete).exists())
            self.assertFalse(dirs["final_maps"].exists())

    def test_scale_bar_uses_nice_projected_length(self) -> None:
        for width in (40.0, 100.0, 800.0, 5000.0):
            length = nice_scale_length(width)
            self.assertGreaterEqual(length / width, 0.15)
            self.assertLessEqual(length / width, 0.25)
            exponent = 10 ** int(np.floor(np.log10(length)))
            self.assertIn(round(length / exponent, 8), (1.0, 2.0, 5.0))
        with self.assertRaises(ValueError):
            nice_scale_length(0)

    def test_font_policy_reports_actual_resolved_family(self) -> None:
        import matplotlib as mpl

        resolved = configure_publication_font()
        self.assertTrue(resolved)
        self.assertEqual(tuple(mpl.rcParams["font.sans-serif"][:3]), PREFERRED_SANS_FONTS)
        self.assertEqual(mpl.rcParams["svg.fonttype"], "none")
        self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)

    def test_stage13_preview_is_optional(self) -> None:
        self.assertNotIn("prediction_preview.png", self.stage13.MAP_REQUIRED_FILES)
        self.assertIn("prediction_preview.png", self.stage13.MAP_OPTIONAL_FILES)

    def test_common_valid_physical_exports_preserve_values_and_grid(self) -> None:
        from rasterio.transform import from_origin
        import rasterio

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "run"
            prepared = output_root / "prepared"
            site = prepared / "SITE"
            site.mkdir(parents=True)
            transform = from_origin(500000.0, 4200000.0, 1.0, 1.0)
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 3,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32652",
                "transform": transform,
                "nodata": float("nan"),
            }
            physical = np.array([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0]], dtype="float32")
            mask = np.array([[1, 0, 1], [0, 1, 1]], dtype="uint8")
            write_raster(site / "depth_200_aligned.tif", physical, profile, nodata=float("nan"))
            write_raster(site / "backscatter_200_aligned.tif", physical * 2, profile, nodata=float("nan"))
            write_raster(
                site / "common_valid_mask.tif",
                mask,
                {**profile, "dtype": "uint8", "nodata": None},
                nodata=None,
                dtype="uint8",
            )
            common = output_root / "common_valid_rasters"
            rows = self.stage01._write_common_valid_site(
                site_id="SITE",
                site_name="Synthetic",
                frequencies=[200],
                prepared_dir=prepared,
                common_valid_root=common,
                output_root=output_root,
            )
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(bool(row["exact_value_equality"]) for row in rows))
            with rasterio.open(common / "SITE/bathymetry_200.tif") as src:
                actual = src.read(1)
                self.assertEqual(src.crs.to_epsg(), 32652)
                self.assertEqual(src.transform, transform)
            np.testing.assert_array_equal(actual[mask == 1], physical[mask == 1])
            self.assertTrue(np.isnan(actual[mask == 0]).all())
            asc = common / "SITE/bathymetry_200.asc"
            txt = common / "SITE/bathymetry_200.txt"
            self.assertEqual(asc.read_bytes(), txt.read_bytes())
            header = asc.read_text(encoding="utf-8").splitlines()[:6]
            self.assertEqual(header[0], "ncols 3")
            self.assertEqual(header[1], "nrows 2")
            self.assertEqual(header[4], "cellsize 1.000000000000")

    def test_reader_facing_terminology_contract(self) -> None:
        text = json.dumps(self.publication, ensure_ascii=False) + (PIPELINE / "13_report_tables_figures.py").read_text(encoding="utf-8")
        for label in (
            "Balanced candidate",
            "Cross-site optimized candidate",
            "Spatial block CV optimized candidate",
            "Dual-frequency transfer candidate",
            "Integrated AP",
            "Predicted habitat extent (ha)",
            "Predicted habitat ratio of valid area (%)",
            "Background reference count",
            "Vegetation reference count",
            "Reference class",
            "Thresholded habitat prediction",
        ):
            self.assertIn(label, text)
        self.assertNotIn("Spatial-CV", text)
        self.assertNotIn("Spatial CV", text)

    def test_composite_axes_are_the_panel_graphical_authority(self) -> None:
        source = (PIPELINE / "13_report_tables_figures.py").read_text(encoding="utf-8")
        for function_name in (
            "_save_study_area_figure",
            "_save_workflow_validation_figure",
            "_save_feature_definition_figure",
            "_main_metric_comparison_figure",
            "_save_candidate_comparison_figure",
            "_save_feature_importance_main",
            "_save_primary_habitat_maps_figure",
        ):
            start = source.index(f"def {function_name}(")
            end = source.find("\ndef ", start + 5)
            body = source[start:] if end < 0 else source[start:end]
            self.assertIn("_export_main_axes(", body, function_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
