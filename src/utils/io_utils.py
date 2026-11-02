from __future__ import annotations

import json
import random


def load_qas(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    queries, answers, qtypes, atypes = [], [], [], []
    for item in data:
        if "question" in item:
            queries.append(item["question"])
        elif "Question" in item:
            queries.append(item["Question"])
        else:
            raise ValueError("No Question found")

        if "answers" in item:
            answers.append(item["answers"])
        elif "answer" in item:
            answers.append(item["answer"])
        elif "Answer" in item:
            answer = []
            for sub_item in item["Answer"]:
                answer_type = sub_item["AnswerType"]
                if answer_type == "Entity":
                    answer.append(sub_item["WikidataLabel"])
                elif answer_type == "Value":
                    answer.append(sub_item["AnswerArgument"])
                else:
                    raise ValueError(f"Unknown Answer Type: {answer_type}")
            answers.append(answer)
        else:
            raise ValueError("No Answer Value")

        qtypes.append(item.get("type") or item.get("question_type"))
        atypes.append(item.get("answer_type"))

    return queries, answers, qtypes, atypes


def maybe_subsample(queries, answers, qtypes, atypes, random_seed: int, sample_size: int | None):
    if sample_size is None or sample_size >= len(queries):
        return queries, answers, qtypes, atypes

    random.seed(random_seed)
    indices = random.sample(range(len(queries)), sample_size)
    return (
        [queries[i] for i in indices],
        [answers[i] for i in indices],
        [qtypes[i] for i in indices],
        [atypes[i] for i in indices],
    )


def read_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return "\n".join(line.strip() for line in f.readlines())
