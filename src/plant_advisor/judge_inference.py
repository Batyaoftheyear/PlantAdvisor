"""Qwen 7B judge on CUDA; model imports happen only inside the GPU command."""

import hashlib
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import (
    append_jsonl, assert_writable_output, fingerprint, load_json, load_jsonl, write_json,
)
from .inference import VERSIONED_PACKAGES, load_model, save_once
from .judge import (
    evaluation_order, export_judgments, judge_messages, load_protocol, load_source,
    parse_evaluation, validate_judgments,
)


def evaluate_one(item, protocol, generate):
    """Bounded JSON repair. All raw responses remain available for inspection."""
    messages = judge_messages(item, protocol)
    attempts = []
    value = None
    for attempt in range(protocol["settings"]["max_retries"] + 1):
        generated = generate(messages)
        raw = generated["raw_output"]
        error = None
        try:
            value = parse_evaluation(raw, item["task"])
        except ValueError as exc:
            error = str(exc)
        attempts.append({**generated, "parse_error": error})
        if error is None:
            break
        if attempt < protocol["settings"]["max_retries"]:
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": protocol["retry_prompt"] + error},
            ])
    return {"evaluation": value, "status": "ok" if value is not None else "invalid_json",
            "attempts": attempts}


def require_sanity(run_dir, manifest, run_id, protocol):
    path = Path(run_dir) / "sanity/judgments.jsonl"
    if not path.exists():
        raise ValueError("Сначала выполните judge без --full: нужны три sanity-оценки")
    rows = load_jsonl(path)
    validate_judgments(rows, manifest, run_id, protocol)
    if ({r["id"] for r in rows} != set(protocol["settings"]["sanity_ids"])
            or any(r["status"] != "ok" for r in rows)):
        raise ValueError("Sanity check не завершён: нужны три корректные JSON-оценки")


def run_judge(config_path, source_dir, questions_path, output_root, *, full=False,
              reference_run=None):
    output_root = assert_writable_output(output_root)
    reference_run = Path(reference_run) if reference_run is not None else None
    protocol = load_protocol(reference_run / "protocol.json" if reference_run else config_path)
    manifest = load_source(source_dir, questions_path)
    inputs = {row["id"]: row for row in manifest["inputs"]}
    if not set(protocol["settings"]["sanity_ids"]) <= inputs.keys():
        raise ValueError("В оцениваемых ответах нет выбранных sanity-примеров")
    if reference_run:
        original_inputs = load_json(reference_run / "input_manifest.json")
        if original_inputs["question_sha256"] != manifest["question_sha256"]:
            raise ValueError("Тест отличается от использованного в эталонном запуске судьи")

    import torch
    from transformers import GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Для судьи нужен CUDA GPU. Выберите T4 в Google Colab")
    settings = protocol["settings"]
    model, tokenizer = load_model(settings)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Модель-судья не загружена в 4-bit")
    generation = GenerationConfig(
        **settings["generation"], bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    )
    versions = {p: importlib.metadata.version(p) for p in VERSIONED_PACKAGES}
    versions["torch"] = str(torch.__version__)
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("judge.py", "judge_inference.py", "inference.py")}
    runtime = {
        "library_versions": versions, "python": sys.version,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "generation_config": generation.to_dict(), "implementation_sha256": implementation,
    }
    if reference_run and load_json(reference_run / "runtime.json") != runtime:
        raise ValueError("Окружение или реализация судьи отличаются от эталонного запуска")
    order = evaluation_order(manifest["inputs"], settings["seed"])
    run_config = {"protocol_sha256": fingerprint(protocol), "input_sha256": fingerprint(manifest),
                  "runtime_sha256": fingerprint(runtime), "evaluation_order": order}
    run_id = fingerprint(run_config)[:16]
    run_dir = output_root / "judge" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("protocol.json", protocol), ("input_manifest.json", manifest),
                        ("runtime.json", runtime), ("evaluation_config.json", run_config)):
        save_once(run_dir / name, value)
    if not (run_dir / "pip_freeze.txt").exists():
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True)
        (run_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    if full:
        require_sanity(run_dir, manifest, run_id, protocol)
    mode = "full" if full else "sanity"
    selected = order if full else settings["sanity_ids"]
    directory = run_dir / mode
    directory.mkdir(exist_ok=True)
    path = directory / "judgments.jsonl"
    rows = load_jsonl(path) if path.exists() else []
    validate_judgments(rows, manifest, run_id, protocol)
    if any(r["id"] not in selected for r in rows):
        raise ValueError("Контрольная точка содержит оценки другого режима")
    completed = {row["id"] for row in rows}

    def generate(messages):
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        input_tokens = encoded["input_ids"].shape[-1]
        if input_tokens > settings["max_input_tokens"]:
            raise ValueError(f"Вход судьи: {input_tokens} токенов, лимит {settings['max_input_tokens']}; текст не обрезан")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=generation)
        tokens = output[0, input_tokens:]
        eos = tokenizer.eos_token_id
        return {"raw_output": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "input_tokens": input_tokens, "output_tokens": len(tokens),
                "finish_reason": "eos" if len(tokens) and tokens[-1].item() == eos else "length"}

    print(f"Судья {settings['model_id']}; {mode}: {len(completed)}/{len(selected)}; {run_dir}", flush=True)
    for key in selected:
        if key in completed:
            continue
        started = time.perf_counter()
        item = inputs[key]
        result = evaluate_one(item, protocol, generate)
        row = {**item, **result, "seconds": round(time.perf_counter() - started, 2),
               "judge_run_id": run_id, "source_run_id": manifest["source_run_id"]}
        append_jsonl(path, row)
        rows.append(row)
        print(f"{key}: {row['status']} {row['evaluation']}", flush=True)
    report = export_judgments(run_dir, mode)
    if not full and manifest["source_run_id"] == "58f7174a5f13e79c":
        known = next(row for row in rows if row["id"] == settings["sanity_error_case"]["id"])
        value = known["evaluation"]
        report["known_error_case"] = settings["sanity_error_case"]
        report["known_error_penalized"] = bool(value and (
            value["factual_correctness"] < 5 or value["actionability"] < 5))
        write_json(directory / "summary.json", report)
        if not report["known_error_penalized"]:
            print("Судья не снизил оценку известного ошибочного ответа. Проверьте объяснения перед полным запуском.", flush=True)
    return {"run_dir": str(run_dir), **report}
