"""CPU fixtures test the pipeline; none of these values are real model scores."""

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from plant_advisor.artifacts import append_jsonl, fingerprint, load_json, write_json
from plant_advisor.judge import (
    EVALUATION_KEYS, evaluation_order, export_judgments, judge_messages, load_protocol,
    load_source, parse_evaluation, summarize_judgments, validate_evaluation, validate_judgments,
)
from plant_advisor.judge_inference import evaluate_one, require_sanity

ROOT = Path(__file__).resolve().parents[1]


def fixture_evaluation(task="diagnosis"):
    return {"factual_correctness": 3, "factual_correctness_explanation": "Тестовое основание.",
            "actionability": 4, "actionability_explanation": "Тестовое основание.",
            "diagnostic_reasoning": 2 if task == "diagnosis" else None,
            "diagnostic_reasoning_explanation": "Тестовое основание.",
            "critical_error": True, "critical_error_explanation": "Тестовое основание."}


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "configs/judge.json")
        self.manifest = load_source(ROOT / "results/baseline", ROOT / "data/eval_questions.jsonl")

    def make_run(self):
        parent = ROOT / "outputs/test-judge"
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / uuid.uuid4().hex
        path.mkdir()

        def cleanup():
            if path.resolve().parent != parent.resolve():
                raise RuntimeError("Некорректный временный каталог")
            shutil.rmtree(path)

        self.addCleanup(cleanup)
        config = {"protocol_sha256": fingerprint(self.protocol),
                  "input_sha256": fingerprint(self.manifest), "runtime_sha256": fingerprint({}),
                  "evaluation_order": evaluation_order(self.manifest["inputs"], 42)}
        for name, value in (("protocol.json", self.protocol), ("input_manifest.json", self.manifest),
                            ("evaluation_config.json", config), ("runtime.json", {})):
            write_json(path / name, value)
        return path, fingerprint(config)[:16]

    def row(self, item, run_id="fixture"):
        evaluation = fixture_evaluation(item["task"])
        return {**item, "evaluation": evaluation, "status": "ok", "seconds": 0.5,
                "source_run_id": self.manifest["source_run_id"], "judge_run_id": run_id,
                "attempts": [{"raw_output": json.dumps(evaluation, ensure_ascii=False),
                              "parse_error": None, "output_tokens": 100,
                              "input_tokens": 200, "finish_reason": "eos"}]}

    def test_scores_and_null(self):
        for task in ("care", "diagnosis"):
            value = fixture_evaluation(task)
            self.assertEqual(parse_evaluation(json.dumps(value), task), value)

    def test_invalid_scores_types_ranges_and_missing_fields(self):
        for field, value in (("factual_correctness", True), ("actionability", "4"),
                             ("actionability", 4.0), ("actionability", 0),
                             ("actionability", 6), ("diagnostic_reasoning", None),
                             ("critical_error", "false"), ("critical_error", 1),
                             ("factual_correctness_explanation", " ")):
            obj = {**fixture_evaluation(), field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_evaluation(obj, "diagnosis")
        obj = fixture_evaluation()
        with self.assertRaises(ValueError):
            validate_evaluation(obj, "care")
        for key in EVALUATION_KEYS:
            obj = fixture_evaluation()
            del obj[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation(obj, "diagnosis")

    def test_malformed_duplicate_and_ambiguous_json(self):
        valid = json.dumps(fixture_evaluation())
        cases = ("", "{}", "[]", valid + valid, "Ответ: " + valid,
                 valid[:-1], valid[:-1] + ', "actionability": 1}',
                 valid.replace('"actionability": 4', '"actionability": NaN'))
        for text in cases:
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                parse_evaluation(text, "diagnosis")
        self.assertEqual(parse_evaluation("```json\n" + valid + "\n```", "diagnosis"),
                         fixture_evaluation())

    def test_prompt_is_blind_and_contains_no_sanity_answer_hints(self):
        item = {**self.manifest["inputs"][0], "model_id": "secret_model", "format_ok": True}
        messages = judge_messages(item, self.protocol)
        self.assertEqual(set(json.loads(messages[1]["content"])), {"task", "question", "answer"})
        self.assertNotIn("secret_model", str(messages))
        self.assertNotIn("format_ok", str(messages))
        self.assertNotIn(self.protocol["settings"]["sanity_error_case"]["issue"], str(messages))

    def test_protocol_roundtrip_and_deterministic_order(self):
        path, _ = self.make_run()
        self.assertEqual(load_protocol(path / "protocol.json"), self.protocol)
        order = evaluation_order(self.manifest["inputs"], 42)
        self.assertEqual(order, evaluation_order(self.manifest["inputs"], 42))
        self.assertEqual(set(order), {r["id"] for r in self.manifest["inputs"]})
        self.assertNotEqual(order, [r["id"] for r in self.manifest["inputs"]])

    def test_json_repair_preserves_raw_attempts(self):
        item = self.manifest["inputs"][0]
        outputs = iter(["broken", json.dumps(fixture_evaluation("care"))])
        prompts = []

        def generate(messages):
            prompts.append(copy.deepcopy(messages))
            return {"raw_output": next(outputs)}

        result = evaluate_one(item, self.protocol, generate)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["raw_output"], "broken")
        self.assertEqual(len(prompts[1]), 4)

    def test_json_failure_is_not_replaced_with_scores(self):
        result = evaluate_one(self.manifest["inputs"][0], self.protocol,
                              lambda messages: {"raw_output": "invalid"})
        self.assertEqual(result["status"], "invalid_json")
        self.assertIsNone(result["evaluation"])
        self.assertEqual(len(result["attempts"]), 2)

    def test_summary_uses_diagnosis_only_and_reports_coverage(self):
        rows = [self.row(item) for item in self.manifest["inputs"]]
        report = summarize_judgments(rows, 30)
        total = report["metrics"][-1]
        self.assertTrue(report["complete"])
        self.assertEqual(total["diagnostic_reasoning_n"], 15)
        self.assertEqual(total["diagnostic_reasoning_mean"], 2)
        self.assertEqual(total["critical_error_rate"], 1)
        rows[0]["status"] = "invalid_json"
        rows[0]["evaluation"] = None
        report = summarize_judgments(rows, 30)
        self.assertFalse(report["complete"])
        self.assertEqual(report["valid"], 29)
        self.assertAlmostEqual(report["coverage"], 29 / 30)
        empty = summarize_judgments([], 30)
        self.assertIsNone(empty["metrics"][-1]["critical_error_rate"])

    def test_checkpoint_rejects_wrong_answer_duplicate_and_edited_scores(self):
        row = self.row(self.manifest["inputs"][0])
        validate_judgments([row], self.manifest, "fixture", self.protocol)
        for field, value in (("answer", "edited"), ("judge_run_id", "foreign"),
                             ("seconds", -1), ("attempts", ["broken"])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_judgments([{**row, field: value}], self.manifest, "fixture", self.protocol)
        with self.assertRaises(ValueError):
            validate_judgments([row, row], self.manifest, "fixture", self.protocol)
        row["evaluation"]["factual_correctness"] = 1
        with self.assertRaises(ValueError):
            validate_judgments([row], self.manifest, "fixture", self.protocol)

    def test_sanity_gate_and_cpu_export(self):
        path, run_id = self.make_run()
        with self.assertRaises(ValueError):
            require_sanity(path, self.manifest, run_id, self.protocol)
        (path / "sanity").mkdir()
        selected = [r for r in self.manifest["inputs"] if r["id"] in self.protocol["settings"]["sanity_ids"]]
        for item in selected:
            append_jsonl(path / "sanity/judgments.jsonl", self.row(item, run_id))
        require_sanity(path, self.manifest, run_id, self.protocol)
        report = export_judgments(path, "sanity")
        self.assertEqual(report["expected_count"], 3)
        self.assertTrue(report["complete"])
        self.assertTrue((path / "sanity/scores.csv").is_file())
        (path / "full").mkdir()
        for item in self.manifest["inputs"]:
            append_jsonl(path / "full/judgments.jsonl", self.row(item, run_id))
        self.assertEqual(export_judgments(path, "full")["valid"], 30)
        result = subprocess.run(
            [sys.executable, "-m", "plant_advisor", "judge-export", "--run-dir", str(path)],
            cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["complete"])
        write_json(path / "runtime.json", {"changed": True})
        with self.assertRaises(ValueError):
            export_judgments(path, "full")

    def test_cpu_imports_cli_and_source_integrity(self):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        script = ("import sys; import plant_advisor.judge; import plant_advisor.judge_inference; "
                  "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                                capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        for args in (["judge", "--help"], ["judge-export", "--help"]):
            result = subprocess.run([sys.executable, "-m", "plant_advisor", *args], env=env,
                                    cwd=ROOT, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.manifest["source_run_id"], "58f7174a5f13e79c")

    def test_judge_notebook_encoding_syntax_and_full_run_disabled(self):
        nb = load_json(ROOT / "notebooks/judge_colab.ipynb")
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("Пробная оценка", text)
        self.assertNotRegex(text, r"\?{2,}|\ufffd")
        values = {}
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            for node in ast.parse("".join(cell["source"]), f"judge-cell-{index}").body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        values[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(values["RUN_FULL_JUDGE"])
        self.assertFalse(values["DOWNLOAD_RESULTS"])


if __name__ == "__main__":
    unittest.main()
