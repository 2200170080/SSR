from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import re
from datetime import datetime


@dataclass
class QAMetrics:
    accuracy: float
    correct_count: int
    total_count: int
    single_accuracy: float | None = None
    multiple_accuracy: float | None = None
    entity_accuracy: float | None = None
    time_accuracy: float | None = None


SINGLE_QTYPES = {"equal", "before_after", "first_last"}
MULTI_QTYPES = {"equal_multi", "before_last", "after_first"}


def hit_rank(answers, results):
    if not results:
        results = [""]
    if results and not isinstance(results[0], str):
        results = [x[0] for x in results]
    if isinstance(answers, list):
        rank = 0
        for result in results:
            rank += 1
            for answer in answers:
                if answer in result:
                    return rank
    elif isinstance(answers, str):
        rank = 0
        for result in results:
            rank += 1
            if answers in result:
                return rank
    return float("inf")


def print_hits(hits, at_n=None):
    if at_n is None:
        at_n = [1, 3, 5, 10, 20, 50, 100]
    total = len(hits)
    for n in at_n:
        cnt = sum(1 for hit in hits if hit <= n)
        print(f"Hit@{n}: {cnt / total:.4f}")


def parse_list_str(llm_answer):
    """
    Normalize LLM output:
    - list -> first element
    - str(list-like) -> first element
    - otherwise -> original string
    """
    if llm_answer is None:
        return ''

    # case 1: already list
    if isinstance(llm_answer, list):
        return str(llm_answer[0]) if llm_answer else ''

    # case 2: not string → cast
    if not isinstance(llm_answer, str):
        return str(llm_answer)

    text = llm_answer.strip()

    # case 3: not list-like string
    if not (text.startswith('[') and text.endswith(']')):
        return text

    # ---- try JSON ----
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except json.JSONDecodeError:
        pass

    # ---- try python literal ----
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
    except Exception:
        pass

    # ---- fallback: tolerant parsing ----
    inner = text[1:-1].strip()

    # split strategy
    if "', '" in inner:
        items = inner.split("', '")
    elif "','" in inner:
        items = inner.split("','")
    else:
        items = re.split(r",\s*", inner)

    # clean quotes
    cleaned = [
        it.strip().strip("'").strip('"')
        for it in items if it.strip()
    ]

    return cleaned[0] if cleaned else ''


def is_correct(answers, llm_answer):
    """
    Compare normalized prediction with ground truth.
    """
    pred = parse_list_str(llm_answer)

    if isinstance(answers, list):
        return pred in answers
    return pred == answers


def evaluate_qa(answers, predictions, qtypes=None, atypes=None) -> QAMetrics:
    total_count = len(predictions)
    correct_count = 0

    single_correct = multiple_correct = entity_correct = time_correct = 0
    total_single = total_multiple = total_entity = total_time = 0

    for idx, (answer, pred) in enumerate(zip(answers, predictions)):
        correct = is_correct(answer, pred)
        if correct:
            correct_count += 1

        if qtypes is not None:
            qtype = qtypes[idx]
            if qtype in SINGLE_QTYPES:
                total_single += 1
                if correct:
                    single_correct += 1
            elif qtype in MULTI_QTYPES:
                total_multiple += 1
                if correct:
                    multiple_correct += 1

        if atypes is not None:
            atype = atypes[idx]
            if atype == "entity":
                total_entity += 1
                if correct:
                    entity_correct += 1
            elif atype == "time":
                total_time += 1
                if correct:
                    time_correct += 1

    return QAMetrics(
        accuracy=correct_count / total_count if total_count else 0.0,
        correct_count=correct_count,
        total_count=total_count,
        single_accuracy=(single_correct / total_single if total_single else None),
        multiple_accuracy=(multiple_correct / total_multiple if total_multiple else None),
        entity_accuracy=(entity_correct / total_entity if total_entity else None),
        time_accuracy=(time_correct / total_time if total_time else None),
    )


def print_qa_metrics(title: str, metrics: QAMetrics) -> None:
    print(f"{title} Accuracy: {metrics.accuracy:.4f} ({metrics.correct_count}/{metrics.total_count})")
    if metrics.single_accuracy is not None:
        print(f"  qtype single:   {metrics.single_accuracy:.5f}")
    if metrics.multiple_accuracy is not None:
        print(f"  qtype multiple: {metrics.multiple_accuracy:.5f}")
    if metrics.entity_accuracy is not None:
        print(f"  atype entity:   {metrics.entity_accuracy:.5f}")
    if metrics.time_accuracy is not None:
        print(f"  atype time:     {metrics.time_accuracy:.5f}")
