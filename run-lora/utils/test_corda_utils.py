import json
import tempfile
import unittest
from pathlib import Path

from corda_utils import clean_task_name, load_corda_texts, normalize_corda_method


class CordaUtilsTest(unittest.TestCase):
    def test_normalize_corda_method_from_peft_name(self):
        self.assertEqual(normalize_corda_method("Corda-KPM"), "kpm")
        self.assertEqual(normalize_corda_method("Corda-IPM"), "ipm")
        self.assertEqual(normalize_corda_method("Corda", "ipm"), "ipm")

    def test_clean_task_name_removes_sample_and_epoch_suffixes(self):
        self.assertEqual(clean_task_name("metamath:100000-ep2"), "metamath")
        self.assertEqual(clean_task_name("python-ep3"), "python")
        self.assertEqual(clean_task_name("commonsense:64"), "commonsense")

    def test_load_local_jsonl_text_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corda.jsonl"
            path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
            self.assertEqual(load_corda_texts(str(path), "metamath", "kpm", 8), ["hello"])

    def test_load_local_task_json_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "metamath"
            task_dir.mkdir()
            (task_dir / "train.json").write_text(
                json.dumps([{"instruction": "Solve 1+1", "output": "2"}]),
                encoding="utf-8",
            )
            self.assertEqual(load_corda_texts(tmp, "metamath:100000", "ipm", 8), ["Solve 1+1\n2"])


if __name__ == "__main__":
    unittest.main()
