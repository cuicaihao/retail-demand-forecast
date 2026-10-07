from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
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

    def test_notebook_code_uses_canonical_paths(self) -> None:
        forbidden = (
            "Path(\"teaching_data\")",
            "Path('prophet_outputs')",
            "Path(\"five_model_outputs\")",
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
        self.assertIn("/artifacts/**", rules)
        self.assertIn("!/artifacts/**/.gitkeep", rules)
        self.assertIn("/data/**", rules)
        self.assertIn("!/data/**/.gitkeep", rules)
        self.assertTrue((ROOT / "artifacts" / "published").is_dir())

    def test_reference_snapshot_provenance_is_explicit(self) -> None:
        data_dir = ROOT / "data" / "teaching"
        prophet_manifest = ROOT / "artifacts/published/prophet/run_manifest.json"
        if not prophet_manifest.is_file():
            self.skipTest("local published artifacts are not present in this clone")
        current = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in data_dir.glob("*.csv")
        }
        manifests = {
            "prophet": prophet_manifest,
            "five_models": ROOT / "artifacts/published/five-model-comparison/comparison_manifest.json",
            "model_series": ROOT / "artifacts/published/model-series/arima/manifest.json",
        }
        hashes = {}
        for name, path in manifests.items():
            record = json.loads(path.read_text())
            hashes[name] = record.get("input_sha256") or record.get("hashes")
        self.assertTrue(all(current[name] == digest for name, digest in hashes["prophet"].items()))
        for snapshot in ("five_models", "model_series"):
            mismatches = {
                name for name, digest in hashes[snapshot].items() if current.get(name) != digest
            }
            self.assertEqual(mismatches, {"wine_scenario_data.csv", "garden_scenario_data.csv"})

    def test_relative_markdown_links_exist(self) -> None:
        link = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
        for markdown in [ROOT / "README.md", *(ROOT / "articles").glob("*.md")]:
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
