"""The original notebook's v2 format metric; semantics are unchanged."""

import re
from statistics import mean

FORMAT_CHECK_VERSION = "v2_standalone_ordered_nonempty_sections"
EXPECTED_SECTIONS = {
    "care": ["условия", "рекомендации"],
    "diagnosis": ["возможные причины", "что проверить", "что делать"],
}


def normalize_heading(line):
    """Recognize standalone headings with the original Markdown normalization."""
    cleaned = line.strip().casefold().replace("ё", "е")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\d{1,2}[.)]\s*", "", cleaned)
    cleaned = re.sub(r"[*_`]", "", cleaned).strip()
    return cleaned.removesuffix(":").strip()


def format_check(text, task):
    if task not in EXPECTED_SECTIONS:
        raise ValueError(f"Неизвестный тип запроса: {task}")
    expected = EXPECTED_SECTIONS[task]
    lines = text.splitlines()
    detected = [(i, normalize_heading(s)) for i, s in enumerate(lines)
                if normalize_heading(s) in expected]
    headings = [h for _, h in detected]
    missing = [h for h in expected if h not in headings]
    duplicates = sorted({h for h in headings if headings.count(h) > 1})
    ordered = headings == expected
    empty = []
    for j, (line_index, heading) in enumerate(detected):
        next_index = detected[j + 1][0] if j + 1 < len(detected) else len(lines)
        body = lines[line_index + 1:next_index]
        if not any(x.strip() for x in body):
            empty.append(heading)
    return {
        "format_ok": not missing and not duplicates and ordered and not empty,
        "section_hits": len(expected) - len(missing),
        "sections_total": len(expected),
        "missing": missing,
        "out_of_order": not ordered and not missing and not duplicates,
        "duplicate_sections": duplicates,
        "empty_sections": empty,
    }


def summarize(records):
    """Same columns and grouping as the baseline, without requiring pandas."""
    if not records:
        raise ValueError("Нет ответов для расчёта метрик")
    groups = [(task, [r for r in records if r["task"] == task])
              for task in sorted({r["task"] for r in records})]
    groups.append(("ALL", records))
    return [
        {"task": task, "count": len(rows),
         "pass_rate": mean(r["format_ok"] for r in rows),
         "avg_section_hits": mean(r["section_hits"] for r in rows),
         "avg_seconds": mean(r["seconds"] for r in rows)}
        for task, rows in groups
    ]
