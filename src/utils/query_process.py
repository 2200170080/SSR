from __future__ import annotations

import asyncio
import json
import re
from typing import Iterable

from tqdm.asyncio import tqdm_asyncio

from utils.io_utils import read_prompt
from utils.eval_utils import evaluate_qa


async def get_query_rewrite(query: str, results: list[str], prompt_path: str, llm_client):
    instruction = read_prompt(prompt_path)
    prompt = instruction.format(query=query, result=results)
    response, _ = await llm_client.request(prompt=prompt, temperature=0, max_tokens=2048, use_cache=False)
    match = re.search(r"<query>(.*?)</query>", response, flags=re.DOTALL)
    return match.group(1).strip() if match else None


async def get_query_graph_pattern(query: str, results: list[str], prompt_path: str, llm_client):
    instruction = read_prompt(prompt_path)
    prompt = instruction.format(question=query, facts=results)

    use_cache = True
    for _ in range(10):
        response, _ = await llm_client.request(
            prompt=prompt,
            temperature=0,
            max_tokens=4096,
            use_cache=use_cache,
        )
        use_cache = False

        fact_match = re.search(r"<fact>(.*?)</fact>", response, flags=re.DOTALL)
        time_match = re.search(r"<time>(.*?)</time>", response, flags=re.DOTALL)
        if fact_match is None or time_match is None:
            continue

        fact_pattern = [x.strip() for x in fact_match.group(1).strip().split(";")]
        time_pattern = [x.strip() for x in time_match.group(1).strip().split(";")]
        if len(fact_pattern) != 3 or len(time_pattern) != 2:
            continue
        return tuple(fact_pattern), tuple(time_pattern)

    return ("?", "?", "?"), ("?", "?")


async def batch_get_query_graph_patterns(
    queries: list[str],
    all_results: list[list[str]],
    prompt_path: str,
    llm_client,
    qa_retrieve_cnt: int,
    concurrency: int,
):
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(query: str, results: list[str]):
        async with semaphore:
            return await get_query_graph_pattern(query, results[:qa_retrieve_cnt], prompt_path, llm_client)

    tasks = [_worker(query, results) for query, results in zip(queries, all_results)]
    return await tqdm_asyncio.gather(*tasks, desc="Extract graph patterns", total=len(tasks))


async def get_qa_result(prompt_text: str, llm_client, retry_prompt_text: str | None = None):
    response, request_ok = await llm_client.request(prompt=prompt_text, temperature=0, max_tokens=4096, use_cache=True)
    answer_pattern = "So the answer is"
    if request_ok and retry_prompt_text is not None and answer_pattern not in response:
        response, retry_ok = await llm_client.request(
            prompt=retry_prompt_text, temperature=0, max_tokens=4096, use_cache=True,
        )
        if not retry_ok:
            return "ERROR", "ERROR: invalid response format"
    if answer_pattern not in response:
        return "ERROR", "ERROR: invalid response format"

    answer = response.split(answer_pattern)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    if answer.startswith(":"):
        answer = answer[1:].strip()
    return answer, response


QUESTION_GRAIN = re.compile('^(?:in\\s+)?(?:which|what)\\s+(year|month)\\b|^(?:can|could)\\s+you\\s+tell\\s+me\\s+(?:the\\s+(?:exact|specific)\\s+|(?:exactly\\s+)?in\\s+which\\s+)(year|month)\\b', re.IGNORECASE)
GRAIN_HINT = {'year': 'In the final "So the answer is:" line, give a supported date as YYYY (four-digit year only). Omit the month and day.', 'month': 'In the final "So the answer is:" line, give a supported date as YYYY-MM (four-digit year and two-digit month). Omit the day.'}


def target_grain(question: str) -> str | None:
    match = QUESTION_GRAIN.match(question.strip())
    return (match[1] or match[2]).lower() if match else None


def with_date_grain(rendered_prompt: str, grain: str | None) -> str:
    if grain is None:
        return rendered_prompt
    return rendered_prompt + "\n" + GRAIN_HINT[grain]


async def batch_run_qa(
    queries: list[str],
    facts_list: list[list[str]],
    prompt_path: str,
    llm_client,
    concurrency: int,
    filters: list[str] | None = None,
):
    instruction = read_prompt(prompt_path)
    instruction, _, retry_instruction = instruction.partition("\n--- SSR Llama format retry ---\n")
    semaphore = asyncio.Semaphore(concurrency)
    filters = filters or [""] * len(queries)

    async def _worker(question: str, facts: list[str], filter_text: str):
        prompt_text = instruction.format(question=question, facts=facts, filter=filter_text)
        retry_prompt_text = (
            retry_instruction.format(question=question, facts=facts, filter=filter_text)
            if retry_instruction else None
        )
        if retry_prompt_text is not None:
            grain = target_grain(question)
            prompt_text = with_date_grain(prompt_text, grain)
            retry_prompt_text = with_date_grain(retry_prompt_text, grain)
        async with semaphore:
            return await get_qa_result(prompt_text, llm_client, retry_prompt_text)

    tasks = [_worker(q, facts, f) for q, facts, f in zip(queries, facts_list, filters)]
    results = await tqdm_asyncio.gather(*tasks, desc="Run QA", total=len(tasks))
    raw_outputs = [x[1] for x in results]
    answers = [x[0] for x in results]
    return answers, raw_outputs
