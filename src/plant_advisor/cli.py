"""CPU-only commands import no torch/transformers dependencies."""

import argparse
import json
import sys
from pathlib import Path

from .artifacts import (
    assert_writable_output, audit_baseline, export_run, load_questions, plot_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PlantAdvisor baseline and artifact checks")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Проверить сохранённый baseline на CPU")
    audit.add_argument("--run-dir", type=Path, default=Path("results/baseline"))
    audit.add_argument("--questions", type=Path, default=Path("data/eval_questions.jsonl"))
    run = commands.add_parser("run", help="GPU: по умолчанию sanity check на 2 вопросах")
    run.add_argument("--config", type=Path, default=Path("results/baseline/config.json"))
    run.add_argument("--questions", type=Path, default=Path("data/eval_questions.jsonl"))
    run.add_argument("--output-root", type=Path, default=Path("outputs"))
    run.add_argument("--full", action="store_true", help="Явно запустить все 30 вопросов")
    run.add_argument("--plot", action="store_true", help="Построить график полного прогона")
    export = commands.add_parser("export", help="CPU: таблицы полного нового прогона")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--questions", type=Path, default=Path("data/eval_questions.jsonl"))
    export.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = audit_baseline(args.questions, args.run_dir)
        elif args.command == "export":
            target = assert_writable_output(args.run_dir)
            result = export_run(target, load_questions(args.questions))
            if args.plot:
                plot_summary(result, target / "baseline_format_rate.png")
        else:
            from .inference import run_baseline
            result = str(run_baseline(args.config, args.questions, args.output_root,
                                      full=args.full, make_plot=args.plot))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0
