"""Release checks for the capability-learning repository."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseValidation(unittest.TestCase):
    def test_readme_has_paper_material_and_reproduction_guide(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for value in (
            "# CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning",
            "## Abstract",
            "assets/overview.png",
            "assets/math-results.png",
            "assets/code-results.png",
            "## Quick Start",
            "yuang-lee/lr-matters-lora",
            "https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html",
            "https://proceedings.neurips.cc/paper_files/paper/2024/hash/83f95bb0ac5046338ea2afe3390e9f4b-Abstract-Conference.html",
            "https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html",
            "https://aclanthology.org/2026.acl-long.890/",
            "https://ojs.aaai.org/index.php/AAAI/article/view/40703",
            "https://icml.cc/virtual/2026/poster/62453",
        ):
            self.assertIn(value, text)
        self.assertNotIn("## Experimental Settings", text)
        self.assertNotIn("## Code Structure", text)

    def test_required_assets_exist(self):
        for path in (
            "assets/overview.png",
            "assets/math-results.png",
            "assets/code-results.png",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_only_representative_scripts_are_published(self):
        expected = {
            f"{task}_{method}.sh"
            for task in ("code", "math")
            for method in (
                "best", "curvlora", "curvdelta", "oplora",
                "replay", "crispedit", "fft", "corda",
            )
        }
        for model in ("gemma", "llama"):
            model_dir = ROOT / "run-lora/scripts" / model
            self.assertEqual({path.name for path in model_dir.glob("*.sh")}, expected)
        self.assertFalse((ROOT / "run-lora/scripts/qwen").exists())

    def test_representative_scripts_keep_lr_sweeps(self):
        scripts = list((ROOT / "run-lora/scripts/gemma").glob("*.sh"))
        scripts += list((ROOT / "run-lora/scripts/llama").glob("*.sh"))
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                assignments = re.findall(r'peft_lrs\["[^"]+"\]="([^"]+)"', text)
                self.assertTrue(assignments)
                self.assertTrue(any(len(values.split()) >= 2 for values in assignments))

    def test_public_method_names_match_the_paper(self):
        blocked = re.compile(
            r"CurvatureLoRA|CurvatureLora|CurvatureDense|CurvatureTrainer|"
            r"curvature_lora|curvature_dense"
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "tests" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".sh", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotRegex(text, blocked)

    def test_root_common_utils_is_the_only_copy(self):
        self.assertTrue((ROOT / "common_utils/server_env.sh").is_file())
        self.assertFalse((ROOT / "run-lora/common_utils").exists())

    def test_public_tree_has_no_credentials_or_private_helpers(self):
        blocked_names = {"feishu_notify.sh", "parse_sftp_profile.py", "local_env.sh"}
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or "tests" in path.parts or not path.is_file():
                continue
            self.assertNotIn(path.name, blocked_names)
            if path.suffix not in {".sh", ".py", ".md", ".json", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}")
                self.assertNotRegex(text, r"wandb_v1_[A-Za-z0-9_-]{20,}")
                self.assertNotRegex(text.lower(), r"feishu|sftp|webhook")
                self.assertNotRegex(text, r"[A-Za-z]:[\\/](?:Users|project)")


if __name__ == "__main__":
    unittest.main()
