from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_only_eight_canonical_top_level_directories_exist(self) -> None:
        expected = {
            "data",
            "docs",
            "images",
            "notebooks",
            "runs",
            "scripts",
            "src",
            "tests",
        }
        actual = {
            path.name
            for path in ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        self.assertEqual(actual, expected)

    def test_canonical_teaching_data_is_unique(self) -> None:
        teaching_dirs = [
            path
            for path in ROOT.rglob("teaching_data")
            if ".venv" not in path.parts and ".git" not in path.parts
        ]
        self.assertEqual(teaching_dirs, [])
        self.assertTrue((ROOT / "data" / "teaching").is_dir())

    def test_source_tree_contains_no_delivery_archives(self) -> None:
        archives = [
            path
            for path in ROOT.rglob("*.zip")
            if ".venv" not in path.parts and "dist" not in path.parts
        ]
        self.assertEqual(archives, [])

    def test_teaching_data_matches_its_manifest(self) -> None:
        data_dir = ROOT / "data" / "teaching"
        if not (data_dir / "generation_metadata.json").is_file():
            self.skipTest("teaching data has not been generated in this clone")
        manifest = json.loads((data_dir / "generation_metadata.json").read_text())
        for name, expected in manifest["csv_sha256"].items():
            actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)
        spec_path = data_dir / "generation_spec.json"
        self.assertEqual(
            manifest.get("generation_spec_sha256"),
            hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        )

    def test_notebook_code_uses_canonical_paths(self) -> None:
        forbidden = (
            'Path("teaching_data")',
            "Path('prophet_outputs')",
            'Path("five_model_outputs")',
        )
        for notebook in (ROOT / "notebooks").glob("*.ipynb"):
            document = json.loads(notebook.read_text())
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in document["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertFalse(any(token in code for token in forbidden), notebook.name)

    def test_notebooks_do_not_store_execution_outputs(self) -> None:
        for notebook in (ROOT / "notebooks").glob("*.ipynb"):
            document = json.loads(notebook.read_text())
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs", []), [], notebook.name)
                    self.assertIsNone(cell.get("execution_count"), notebook.name)

    def test_generated_artifacts_are_ignored_by_policy(self) -> None:
        rules = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("/data/**", rules)
        self.assertIn("!/data/**/.gitkeep", rules)
        self.assertTrue((ROOT / "data" / "artifacts" / "published").is_dir())

    def test_benchmark_store_separates_protocols_releases_and_history(self) -> None:
        benchmark_dir = ROOT / "data" / "benchmarks"
        self.assertFalse((benchmark_dir / "results").exists())
        self.assertTrue((benchmark_dir / "protocols").is_dir())
        self.assertTrue((benchmark_dir / "published").is_dir())
        self.assertTrue((benchmark_dir / "archive_single_cutoff_28d").is_dir())

    def test_published_benchmark_release_hashes_recompute(self) -> None:
        published = ROOT / "data" / "benchmarks" / "published"
        for release_path in published.glob("**/release_manifest.json"):
            release = json.loads(release_path.read_text())
            release_dir = release_path.parent
            for relative, expected in release["files"].items():
                path = release_dir / relative
                self.assertTrue(path.is_file(), relative)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), expected
                )

    def test_current_v4_report_matches_timesfm_release_provenance(self) -> None:
        release_id = "20261106-timesfm-v1"
        expanding = (
            ROOT
            / "data/benchmarks/published/2026-retail-week-expanding-v4/releases"
            / release_id
        )
        comparison = (
            ROOT
            / "data/benchmarks/published/comparisons"
            / "2026-retail-week-expanding-v4--vs--2026-retail-week-104w-rolling-v4"
            / "releases"
            / release_id
        )
        report = (
            ROOT / "docs/reports/2026-retail-week-v4-first-stage-report.md"
        ).read_text()
        expanding_manifest = expanding / "release_manifest.json"
        comparison_manifest = comparison / "release_manifest.json"
        release = json.loads(expanding_manifest.read_text())

        self.assertEqual(len(release["model_ids"]), 41)
        self.assertIn(release_id, report)
        self.assertIn(
            hashlib.sha256(expanding_manifest.read_bytes()).hexdigest(), report
        )
        self.assertIn(
            hashlib.sha256(comparison_manifest.read_bytes()).hexdigest(), report
        )
        self.assertIn("TimesFM 3.0 + Known Covariates", report)

    def test_reference_snapshot_provenance_is_explicit(self) -> None:
        data_dir = ROOT / "data" / "teaching"
        prophet_manifest = ROOT / "data/artifacts/published/prophet/run_manifest.json"
        if not prophet_manifest.is_file():
            self.skipTest("local published artifacts are not present in this clone")
        current = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in data_dir.glob("*.csv")
        }
        manifests = {
            "prophet": prophet_manifest,
            "five_models": (
                ROOT
                / "data/artifacts/published/five-model-comparison/comparison_manifest.json"
            ),
            "model_series": (
                ROOT / "data/artifacts/published/model-series/arima/manifest.json"
            ),
        }
        hashes = {}
        for name, path in manifests.items():
            record = json.loads(path.read_text())
            hashes[name] = record.get("input_sha256") or record.get("hashes")
        expected_mismatches = {
            "prophet": {
                "grocery_scenario_data.csv",
                "wine_scenario_data.csv",
                "garden_scenario_data.csv",
                "vic_holidays.csv",
            },
            "five_models": {
                "grocery_scenario_data.csv",
                "wine_scenario_data.csv",
                "garden_scenario_data.csv",
                "vic_holidays.csv",
            },
            "model_series": {
                "grocery_scenario_data.csv",
                "wine_scenario_data.csv",
                "garden_scenario_data.csv",
                "vic_holidays.csv",
                "scenario_diagnostics.csv",
            },
        }
        for snapshot in ("prophet", "five_models", "model_series"):
            mismatches = {
                name
                for name, digest in hashes[snapshot].items()
                if current.get(name) != digest
            }
            self.assertEqual(
                mismatches,
                expected_mismatches[snapshot],
            )

    def test_relative_markdown_links_exist(self) -> None:
        link = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
        for markdown in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
            for target in link.findall(markdown.read_text()):
                if target.startswith(("http://", "https://", "#")):
                    continue
                path_text = target.split("#", 1)[0]
                if path_text:
                    resolved = (markdown.parent / path_text).resolve()
                    if "artifacts" in resolved.parts:
                        continue
                    self.assertTrue(
                        resolved.exists(),
                        (markdown, target),
                    )


if __name__ == "__main__":
    unittest.main()
