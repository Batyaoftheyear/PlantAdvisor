"""Read, validate, fingerprint and export experiment artifacts on CPU."""

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .metrics import EXPECTED_SECTIONS, FORMAT_CHECK_VERSION, format_check, summarize


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: повреждён JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: ожидался JSON-объект")
        rows.append(row)
    return rows


def append_jsonl(path, row):
    """Append to a validated checkpoint, including one without a trailing newline."""
    target = assert_writable_output(path)
    with target.open("ab+") as stream:
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write((canonical_json(row) + "\n").encode("utf-8"))


def load_questions(path, expected_count=30):
    rows = load_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"Ожидалось {expected_count} вопросов, получено {len(rows)}")
    seen = set()
    for row in rows:
        if (not isinstance(row.get("id"), str) or not row["id"].strip()
                or row["id"] in seen):
            raise ValueError("Пустой или дублированный ID вопроса")
        seen.add(row["id"])
        if row.get("task") not in EXPECTED_SECTIONS:
            raise ValueError(f"Неизвестный тип вопроса: {row.get('task')}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Пустой вопрос: {row['id']}")
    return rows


def validate_config(config, questions):
    if config.get("question_sha256") != fingerprint(questions):
        raise ValueError("Хеш тестовых вопросов не совпадает с конфигурацией")
    if config.get("format_check_version") != FORMAT_CHECK_VERSION:
        raise ValueError("Неподдерживаемая версия форматной метрики")
    for field in ("experiment", "model_id", "model_revision", "system_prompt"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"Не задано поле конфигурации: {field}")
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
        raise ValueError("seed должен быть целым числом")
    if type(config.get("max_new_tokens")) is not int or config["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens должен быть положительным целым")
    if config.get("decoding") != {"do_sample": False, "num_beams": 1}:
        raise ValueError("Этот baseline поддерживает только greedy decoding")
    if config.get("quantization") != {
        "bits": 4, "type": "nf4", "double_quant": True, "compute_dtype": "float16"
    }:
        raise ValueError("Этот baseline поддерживает только исходную 4-bit NF4 конфигурацию")
    versions = config.get("library_versions")
    if not isinstance(versions, dict) or not versions:
        raise ValueError("Не сохранены версии библиотек")
    return fingerprint(config)[:16]


def validate_records(records, questions, config, *, require_complete=True):
    """Validate complete baselines or partial checkpoints before resuming."""
    run_id = validate_config(config, questions)
    expected = {q["id"]: q for q in questions}
    seen = set()
    for row in records:
        key = row.get("id")
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError(f"Неожиданный или дублированный ID ответа: {key}")
        seen.add(key)
        source = expected[key]
        for field in ("question", "task"):
            if row.get(field) != source[field]:
                raise ValueError(f"{key}: несовпадение поля {field}")
        for field, value in (("run_id", run_id), ("model_id", config["model_id"]),
                             ("model_revision", config["model_revision"])):
            if row.get(field) != value:
                raise ValueError(f"{key}: несовпадение поля {field}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"{key}: ответ должен быть строкой")
        seconds = row.get("seconds")
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"{key}: недопустимое время генерации")
        for field, value in format_check(row["answer"], row["task"]).items():
            if type(row.get(field)) is not type(value) or row[field] != value:
                raise ValueError(f"{key}: сохранённая метрика {field} не совпадает")
    if require_complete and seen != set(expected):
        raise ValueError(f"Неполный прогон: {len(seen)}/{len(expected)} ответов")
    return run_id


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_answer_csv(path, records):
    rows = read_csv(path)
    expected = {r["id"]: r for r in records}
    seen = set()
    for row in rows:
        key = row.get("id")
        if key not in expected or key in seen:
            raise ValueError("CSV ответов содержит неизвестный или повторный ID")
        seen.add(key)
        if set(row) != set(expected[key]):
            raise ValueError(f"CSV ответов: несовпадение колонок для {key}")
        for field, value in expected[key].items():
            actual = row[field]
            if isinstance(value, (list, bool)):
                try:
                    actual = ast.literal_eval(actual)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"CSV ответов: неверное поле {field}") from exc
            elif isinstance(value, (int, float)):
                actual = float(actual)
            if actual != value:
                raise ValueError(f"CSV ответов: несовпадение {key}.{field}")
    if seen != set(expected):
        raise ValueError("CSV ответов неполон")


def validate_metrics_csv(path, summary):
    rows = read_csv(path)
    expected = {row["task"]: row for row in summary}
    if Counter(row.get("task") for row in rows) != Counter(expected.keys()):
        raise ValueError("CSV метрик: несовпадение групп")
    for row in rows:
        for field, value in expected[row["task"]].items():
            if field == "task":
                continue
            if not math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"CSV метрик: несовпадение {row['task']}.{field}")


def audit_baseline(questions_path, run_dir):
    run_dir = Path(run_dir)
    questions = load_questions(questions_path)
    config = load_json(run_dir / "config.json")
    records = load_jsonl(run_dir / "baseline_answers.jsonl")
    run_id = validate_records(records, questions, config)
    summary = summarize(records)
    validate_answer_csv(run_dir / "baseline_answers.csv", records)
    validate_metrics_csv(run_dir / "baseline_metrics.csv", summary)
    return {"run_id": run_id, "question_count": len(questions),
            "answer_count": len(records), "task_counts": dict(Counter(q["task"] for q in questions)),
            "failed_ids": [r["id"] for r in records if not r["format_ok"]],
            "metrics": summary}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_run(run_dir, questions):
    """Export only a complete, validated run. No ML dependencies on CPU."""
    run_dir = Path(run_dir)
    assert_writable_output(run_dir)
    config = load_json(run_dir / "config.json")
    records = load_jsonl(run_dir / "baseline_answers.jsonl")
    validate_records(records, questions, config)
    records = sorted(records, key=lambda r: r["id"])
    summary = summarize(records)
    write_csv(run_dir / "baseline_answers.csv", records)
    write_csv(run_dir / "baseline_metrics.csv", summary)
    return summary


def assert_writable_output(path):
    """Keep new results outside tracked baseline/reference/data locations."""
    target = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    for protected in (repo / "results", repo / "data", repo / "notebooks"):
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError("Новые результаты должны находиться вне results/, data/ и notebooks/")
    if target == repo:
        raise ValueError("Нельзя сохранять результаты в корне репозитория")
    return target


def plot_summary(summary, path):
    """Optional plotting dependency, imported only when requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar([r["task"] for r in summary], [r["pass_rate"] for r in summary])
    ax.set(ylim=(0, 1), ylabel="Доля ответов, прошедших проверку",
           title="Baseline: соблюдение структуры по типам запросов")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
