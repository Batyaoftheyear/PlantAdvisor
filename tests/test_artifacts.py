import ast
import copy
import hashlib
from contextlib import contextmanager
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from plant_advisor.artifacts import (
    append_jsonl, assert_writable_output, audit_baseline, export_run, fingerprint, load_json,
    load_jsonl, load_questions, validate_config, validate_records, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/baseline"
QUESTIONS = ROOT / "data/eval_questions.jsonl"


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = ROOT / "outputs/test-tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.questions = load_questions(QUESTIONS)
        self.config = load_json(BASELINE / "config.json")
        self.records = load_jsonl(BASELINE / "baseline_answers.jsonl")

    @contextmanager
    def temporary_directory(self):
        # Inherit workspace ACLs: Python 3.14's mode=0700 temp directories are
        # inaccessible to child operations in this managed Windows sandbox.
        target = self.temp_root / uuid.uuid4().hex
        target.mkdir()
        try:
            yield target
        finally:
            if target.resolve().parent != self.temp_root.resolve():
                raise RuntimeError("Временный каталог вышел за пределы рабочей папки")
            shutil.rmtree(target)

    def test_real_baseline(self):
        audit = audit_baseline(QUESTIONS, BASELINE)
        self.assertEqual(audit["run_id"], "58f7174a5f13e79c")
        self.assertEqual(audit["question_count"], 30)
        self.assertEqual(audit["answer_count"], 30)
        self.assertEqual(audit["task_counts"], {"care": 15, "diagnosis": 15})
        self.assertEqual(audit["failed_ids"], ["q04", "q16", "q18", "q26"])
        self.assertAlmostEqual(audit["metrics"][-1]["pass_rate"], 26 / 30)

    def test_frozen_inputs_have_original_bytes(self):
        manifest = load_json(ROOT / "docs/stage1_input_sha256.json")
        for name, expected in manifest.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_config_fingerprints(self):
        self.assertEqual(fingerprint(self.questions), self.config["question_sha256"])
        self.assertEqual(validate_config(self.config, self.questions), "58f7174a5f13e79c")

    def test_reference_notebook_constants_match_artifacts(self):
        nb = load_json(ROOT / "notebooks/reference/01_baseline_colab.ipynb")
        values = {}
        for cell in nb["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] != "code" or source.lstrip().startswith("%"):
                continue
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        values[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertEqual(values["EVAL_QUESTIONS"], self.questions)
        for name, key in (("MODEL_ID", "model_id"), ("SYSTEM_PROMPT", "system_prompt"),
                          ("MAX_NEW_TOKENS", "max_new_tokens"), ("SEED", "seed"),
                          ("FORMAT_CHECK_VERSION", "format_check_version")):
            self.assertEqual(values[name], self.config[key])

    def test_runner_notebook_syntax_and_default_switches(self):
        nb = load_json(ROOT / "notebooks/baseline_colab.ipynb")
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("Запуск в Google Colab на GPU T4", text)
        self.assertNotRegex(text, r"\?{2,}|\ufffd")
        values = {}
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            tree = ast.parse("".join(cell["source"]), filename=f"runner-cell-{index}")
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        values[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(values["RUN_FULL_EVAL"])
        self.assertTrue(values["RUN_SANITY"])
        self.assertFalse(values["DOWNLOAD_RESULTS"])

    def test_changed_question_hash(self):
        self.questions[0]["question"] += "?"
        with self.assertRaisesRegex(ValueError, "Хеш"):
            validate_config(self.config, self.questions)

    def test_config_changes_cannot_reuse_old_answers(self):
        self.config["seed"] += 1
        with self.assertRaisesRegex(ValueError, "run_id"):
            validate_records(self.records, self.questions, self.config)

    def test_unsupported_config_is_rejected(self):
        for field, value in (("decoding", {"do_sample": True, "num_beams": 1}),
                             ("format_check_version", "v3"),
                             ("max_new_tokens", 0), ("quantization", {"bits": 8})):
            config = {**self.config, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(config, self.questions)

    def test_partial_records_only_allowed_for_resume(self):
        for records in ([], self.records[:2]):
            with self.subTest(count=len(records)):
                validate_records(records, self.questions, self.config, require_complete=False)
                with self.assertRaisesRegex(ValueError, "Неполный"):
                    validate_records(records, self.questions, self.config)

    def test_checkpoint_corruption(self):
        mutations = {
            "question": "другой вопрос", "task": "diagnosis", "run_id": "foreign",
            "model_id": "foreign", "model_revision": "foreign", "id": "q99",
            "format_ok": False, "section_hits": 0, "seconds": -1, "answer": None,
        }
        for field, value in mutations.items():
            records = copy.deepcopy(self.records)
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records(records, self.questions, self.config)

    def test_nonfinite_seconds(self):
        for value in (float("nan"), float("inf"), True):
            records = copy.deepcopy(self.records)
            records[0]["seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_records(records, self.questions, self.config)

    def test_duplicate_answers(self):
        records = self.records + [self.records[0]]
        with self.assertRaisesRegex(ValueError, "дублированный"):
            validate_records(records, self.questions, self.config)

    def test_append_resumes_after_missing_final_newline(self):
        with self.temporary_directory() as temp:
            path = temp / "answers.jsonl"
            path.write_text(json.dumps(self.records[0], ensure_ascii=False), encoding="utf-8")
            append_jsonl(path, self.records[1])
            self.assertEqual(load_jsonl(path), self.records[:2])
            validate_records(load_jsonl(path), self.questions, self.config, require_complete=False)

    def test_question_validation_and_jsonl_errors(self):
        with self.temporary_directory() as temp:
            path = Path(temp) / "questions.jsonl"
            for content in ("{bad json}\n", "[]\n"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_jsonl(path)
            path.write_text("\n" + json.dumps(self.questions[0], ensure_ascii=False) + "\n\n",
                            encoding="utf-8")
            self.assertEqual(load_questions(path, expected_count=1), self.questions[:1])
            with self.assertRaises(ValueError):
                load_questions(path)
            for rows in ([self.questions[0], self.questions[0]],
                         [{**self.questions[0], "task": "other"}],
                         [{**self.questions[0], "question": " "}]):
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    load_questions(path, expected_count=len(rows))

    def test_corrupt_csv_and_metrics_are_detected(self):
        with self.temporary_directory() as temp:
            target = Path(temp) / "run"
            shutil.copytree(BASELINE, target)
            for filename in ("baseline_answers.csv", "baseline_metrics.csv"):
                path = target / filename
                original = path.read_bytes()
                path.write_text("task,count\nALL,0\n", encoding="utf-8")
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    audit_baseline(QUESTIONS, target)
                path.write_bytes(original)

    def test_export_roundtrip(self):
        with self.temporary_directory() as temp:
            target = Path(temp) / "new-run"
            target.mkdir()
            write_json(target / "config.json", self.config)
            shutil.copyfile(BASELINE / "baseline_answers.jsonl", target / "baseline_answers.jsonl")
            export_run(target, self.questions)
            self.assertEqual(audit_baseline(QUESTIONS, target), audit_baseline(QUESTIONS, BASELINE))

    def test_export_rejects_partial_run(self):
        with self.temporary_directory() as temp:
            target = Path(temp)
            write_json(target / "config.json", self.config)
            (target / "baseline_answers.jsonl").write_text(
                json.dumps(self.records[0], ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Неполный"):
                export_run(target, self.questions)
            self.assertFalse((target / "baseline_metrics.csv").exists())

    def test_reference_and_data_output_protection(self):
        for path in (ROOT, ROOT.parent, BASELINE, BASELINE / "runs", ROOT / "data",
                     ROOT / "notebooks", ROOT / "results/new-run"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                assert_writable_output(path)
        self.assertEqual(assert_writable_output(ROOT / "outputs"), ROOT / "outputs")
        with self.assertRaises(ValueError):
            export_run(BASELINE, self.questions)

    def test_cpu_imports_do_not_load_ml_packages(self):
        script = ("import sys; import plant_advisor.cli; import plant_advisor.inference; "
                  "assert not any(n in sys.modules for n in "
                  "('torch', 'transformers', 'matplotlib', 'pandas', 'bitsandbytes'))")
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_audit_help_and_output_guard(self):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        for args in (["audit"], ["--help"], ["run", "--help"]):
            result = subprocess.run([sys.executable, "-m", "plant_advisor", *args],
                                    cwd=ROOT, env=env, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            if args == ["audit"]:
                self.assertEqual(json.loads(result.stdout)["run_id"], "58f7174a5f13e79c")
        result = subprocess.run(
            [sys.executable, "-m", "plant_advisor", "run", "--output-root", "results/baseline"],
            cwd=ROOT, env=env, capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Новые результаты", result.stderr)


if __name__ == "__main__":
    unittest.main()
