import ast
import json
import re
import unittest
from pathlib import Path

from plant_advisor.artifacts import load_jsonl
from plant_advisor.metrics import format_check, normalize_heading, summarize

ROOT = Path(__file__).resolve().parents[1]
CARE = "Условия\nСвет.\nРекомендации\nПроверять грунт."
DIAGNOSIS = "Возможные причины\nСтресс.\nЧто проверить\nУсловия.\nЧто делать\nНаблюдать."


class FormatCheckTests(unittest.TestCase):
    def test_valid_styles(self):
        for headings in (("Условия", "Рекомендации"),
                         ("### Условия", "### Рекомендации"),
                         ("**Условия:**", "**Рекомендации:**"),
                         ("1. Условия", "2) Рекомендации"),
                         ("`УСЛОВИЯ:`", "_РЕКОМЕНДАЦИИ_")):
            with self.subTest(headings=headings):
                self.assertTrue(format_check(f"{headings[0]}\nСвет.\n{headings[1]}\nПолив.",
                                             "care")["format_ok"])
        self.assertTrue(format_check(DIAGNOSIS, "diagnosis")["format_ok"])

    def test_additional_sections_are_allowed(self):
        text = "Вступление\n" + CARE + "\nДополнительно\nНаблюдайте."
        self.assertTrue(format_check(text, "care")["format_ok"])

    def test_inline_headings_do_not_count(self):
        result = format_check("Условия: свет.\nРекомендации: полив.", "care")
        self.assertFalse(result["format_ok"])
        self.assertEqual(result["section_hits"], 0)

    def test_empty_answer_and_empty_sections(self):
        for text in ("", " \n\t", "Условия\n\nРекомендации\nПолив.",
                     "Условия\nСвет.\nРекомендации\n \t"):
            with self.subTest(text=text):
                self.assertFalse(format_check(text, "care")["format_ok"])

    def test_duplicate_heading(self):
        result = format_check(CARE + "\nРекомендации\nЕщё.", "care")
        self.assertFalse(result["format_ok"])
        self.assertEqual(result["duplicate_sections"], ["рекомендации"])

    def test_reordered_sections(self):
        result = format_check("Рекомендации\nПолив.\nУсловия\nСвет.", "care")
        self.assertTrue(result["out_of_order"])
        self.assertEqual(result["missing"], [])

    def test_missing_section_is_not_marked_as_order_error(self):
        result = format_check("Что проверить\nСвет.\nЧто делать\nНаблюдать.", "diagnosis")
        self.assertEqual(result["missing"], ["возможные причины"])
        self.assertFalse(result["out_of_order"])

    def test_unknown_task(self):
        with self.assertRaises(ValueError):
            format_check(CARE, "other")

    def test_existing_metric_weakness_is_preserved(self):
        # v2 considers any non-whitespace line content; tightening it would change the metric.
        self.assertTrue(format_check("Условия\n-\nРекомендации\n### Дополнительно", "care")["format_ok"])

    def test_normalization(self):
        self.assertEqual(normalize_heading(" ### 1. **УСЛОВИЯ:** "), "условия")
        self.assertEqual(normalize_heading("Ёлка:"), "елка")

    def test_reference_notebook_parity(self):
        nb = json.loads((ROOT / "notebooks/reference/01_baseline_colab.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "code" and "def format_check(" in "".join(c["source"]))
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
        scope = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "reference_metric", "exec"), scope)
        cases = [(r["answer"], r["task"]) for r in
                 load_jsonl(ROOT / "results/baseline/baseline_answers.jsonl")]
        cases += [(CARE, "care"), (DIAGNOSIS, "diagnosis"), ("", "care"),
                  (CARE + "\nУсловия\nЕщё", "care"),
                  ("Условия\n-\nРекомендации\n### Иное", "care"),
                  ("Условия: свет.\nРекомендации: полив.", "care")]
        for text, task in cases:
            with self.subTest(task=task, text=text[:30]):
                self.assertEqual(format_check(text, task), scope["format_check"](text, task))

    def test_summary_requires_answers(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
