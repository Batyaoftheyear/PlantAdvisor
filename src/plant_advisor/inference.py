"""Optional CUDA baseline, using the saved configuration as the source of truth."""

import hashlib
import importlib.metadata
import random
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import (
    append_jsonl, assert_writable_output, canonical_json, export_run, load_json,
    load_jsonl, load_questions, plot_summary, validate_config, validate_records, write_json,
)
from .metrics import format_check

VERSIONED_PACKAGES = (
    "transformers", "accelerate", "bitsandbytes", "huggingface-hub", "safetensors",
)


def load_model(config):
    """Same quantization/chat template/decoding as the reference notebook."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Нужен CUDA GPU; в Colab выберите T4")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["type"],
        bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"], device_map="auto",
        quantization_config=quantization, torch_dtype=torch.float16,
    )
    model.eval()
    actual_commit = getattr(model.config, "_commit_hash", None)
    if actual_commit != config["model_revision"]:
        raise RuntimeError(f"Ревизия загруженной модели не совпадает: {actual_commit}")
    return model, tokenizer


def answer_question(question, config, model, tokenizer):
    import torch

    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": question},
    ]
    with torch.inference_mode():
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        ).to("cuda")
        output = model.generate(
            inputs, max_new_tokens=config["max_new_tokens"], **config["decoding"],
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def save_once(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Конфликт сохранённой конфигурации: {path}")
    else:
        write_json(path, value)


def run_baseline(config_path, questions_path, output_root, *, full=False, make_plot=False):
    # Read-only validation and output protection happen before importing ML packages.
    output_root = assert_writable_output(output_root)
    questions = load_questions(questions_path)
    source_config = load_json(config_path)
    source_id = validate_config(source_config, questions)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Этот шаг запускается на Colab T4")
    versions = {name: importlib.metadata.version(name) for name in VERSIONED_PACKAGES}
    versions["torch"] = str(torch.__version__)
    config = {**source_config, "library_versions": versions}
    run_id = validate_config(config, questions)
    if versions != source_config["library_versions"]:
        print("Версии отличаются от исходного baseline; новый run_id:", run_id, flush=True)
        print("Исходные:", source_config["library_versions"], flush=True)
        print("Текущие:", versions, flush=True)
    run_dir = output_root / ("runs" if full else "sanity") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_once(run_dir / "config.json", config)
    save_once(run_dir / "source_config.json", source_config)
    path = run_dir / "baseline_answers.jsonl"
    records = load_jsonl(path) if path.exists() else []
    validate_records(records, questions, config, require_complete=False)
    selected = questions if full else questions[:2]
    if any(r["id"] not in {q["id"] for q in selected} for r in records):
        raise ValueError("Контрольная точка содержит ответы вне выбранного режима")

    model, tokenizer = load_model(config)
    runtime = {
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "torch": str(torch.__version__), "python": sys.version,
        "model_revision": config["model_revision"],
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "generation_config": model.generation_config.to_dict(),
        "source_run_id": source_id, "mode": "full" if full else "sanity",
        "inference_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    # A changed CUDA/GPU/model defaults/code must not silently resume the same file.
    save_once(run_dir / "runtime.json", runtime)
    if not (run_dir / "pip_freeze.txt").exists():
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True)
        (run_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    questions_copy = run_dir / "eval_questions.jsonl"
    if questions_copy.exists():
        if load_jsonl(questions_copy) != questions:
            raise ValueError("Несовпадение сохранённой копии вопросов")
    else:
        questions_copy.write_text("".join(canonical_json(q) + "\n" for q in questions),
                                  encoding="utf-8")

    completed = {r["id"] for r in records}
    print(f"{run_dir}: готово {len(completed)}/{len(selected)}", flush=True)
    for item in selected:
        if item["id"] in completed:
            continue
        start = time.perf_counter()
        response = answer_question(item["question"], config, model, tokenizer)
        row = {**item, "answer": response, "seconds": round(time.perf_counter() - start, 2),
               **format_check(response, item["task"]), "run_id": run_id,
               "model_id": config["model_id"], "model_revision": config["model_revision"]}
        append_jsonl(path, row)
        records.append(row)
        print(f"{item['id']}: format_ok={row['format_ok']}\n{response}\n", flush=True)
    if full:
        summary = export_run(run_dir, questions)
        if make_plot:
            plot_summary(summary, run_dir / "baseline_format_rate.png")
    else:
        print("Sanity check: 2 вопроса. Итоговые метрики полного теста не формируются.", flush=True)
    return run_dir
