import os
import unittest

import eval_world_knowledge as wk


class WorldKnowledgeHelpersTest(unittest.TestCase):
    def test_extract_scores_accepts_lm_eval_filter_suffixes(self):
        results = {
            "results": {
                "triviaqa": {"exact_match,remove_whitespace": 0.12345},
                "nq_open": {"exact_match,remove_whitespace": 0.23456},
                "webqs": {"f1,none": 0.34567},
            }
        }

        scores, details = wk.extract_world_knowledge_scores(results, wk.TASKS)

        self.assertEqual(
            scores,
            {"triviaqa": 0.1235, "nq_open": 0.2346, "webqs": 0.3457},
        )
        self.assertEqual(
            details["selected_metrics"],
            {
                "triviaqa": "exact_match,remove_whitespace",
                "nq_open": "exact_match,remove_whitespace",
                "webqs": "f1,none",
            },
        )

    def test_lora_ga_base_model_path_includes_task(self):
        base = wk.resolve_base_model(
            peft="LoRA-GA",
            output_home="/exp",
            model_abbr="Qwen3-0.6B",
            rank=128,
            output_path="/unused",
            model="/base",
            timestamp="20260710-120000",
            task="metamath",
        )

        self.assertTrue(
            os.path.normpath(base).endswith(
                os.path.normpath("/exp/output/LoRA-GA-Qwen3-0.6B-r128-metamath")
            )
        )

    def test_corda_base_model_path_includes_method_and_task(self):
        base = wk.resolve_base_model(
            peft="Corda-IPM",
            output_home="/exp",
            model_abbr="Qwen3-0.6B",
            rank=128,
            output_path="/unused",
            model="/base",
            timestamp="20260710-120000",
            task="metamath",
        )

        self.assertTrue(
            os.path.normpath(base).endswith(
                os.path.normpath("/exp/output/Corda-IPM-Qwen3-0.6B-r128-metamath")
            )
        )

    def test_corda_kpm_base_model_path_includes_task(self):
        base = wk.resolve_base_model(
            peft="Corda-KPM",
            output_home="/exp",
            model_abbr="Llama-2-7b",
            rank=32,
            output_path="/unused",
            model="/base",
            timestamp="20260710-120000",
            task="python",
        )

        self.assertTrue(
            os.path.normpath(base).endswith(
                os.path.normpath("/exp/output/Corda-KPM-Llama-2-7b-r32-python")
            )
        )


if __name__ == "__main__":
    unittest.main()
