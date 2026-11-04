import asyncio
import ast
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("src").resolve()))
from tqdm.asyncio import tqdm_asyncio
from run_pipeline import build_graph_retriever
from utils.client import AsyncLLMClient
from utils.eval_utils import is_correct
from utils.io_utils import load_qas, read_prompt
from utils.query_process import get_qa_result
from utils.sample_utils import uniform_sample_preserve_edges

out = Path(os.environ["GP_CONTRACT_RUN"])
source = Path("artifacts/gp-B-full-MqkyPV/llm_cache.jsonl")
gp_text = read_prompt("artifacts/gp-B-full-MqkyPV/gp-B.txt")
assert source.is_file()

qs, answers, _, kinds = load_qas("datasets/MultiTQ/questions/test.json")
frequency = Counter(qs)
gold = {
    q: (a, k) for q, a, k in zip(qs, answers, kinds)
    if frequency[q] == 1
}
excluded = set()

def collect(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"question", "Question", "题目"} and isinstance(item, str):
                excluded.add(item)
            else:
                collect(item)
    elif isinstance(value, list):
        for item in value:
            collect(item)

for name in [
    "gp-wildcard-ab/questions.json",
    "gp-holdout-2eEr22/questions.json",
    "gp-role-ab-Paj2eX/questions.json",
    "qa-order-ab-34VEXw/jobs.json",
    "manual-pattern-diag-2M9xVw/report.json",
    "qa-entity-label-G85HfM/cases.json",
    "qa-label-holdout-6Uy1Kh/jobs.json",
]:
    collect(json.loads(
        (Path("artifacts") / name).read_text(encoding="utf-8")
    ))

excluded.update(qs[i] for i in random.Random(42).sample(range(len(qs)), 200))
excluded.add(
    "Before China, which country was the last to receive an appeal "
    "from Thailand to the Office of the Attorney General?"
)

qa_re = re.compile(
    r"(?m)^Question: ([^\n]*)\nFacts filter condition: ([^\n]*)"
    r"\nHistorical facts:\n(.*?)\nAnswer:", re.S
)
pool = {}
print("扫描缓存，按模式形状选题，不依据原答案对错选题……", flush=True)

with source.open(encoding="utf-8") as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        if row["input"][1] != "qwen14b":
            continue
        found = list(qa_re.finditer(row["input"][0]))
        if not found:
            continue
        q, condition, _ = found[-1].groups()
        if q not in gold or gold[q][1] not in {"entity", "time"}:
            continue
        if gold[q][1] == "entity" and q in excluded:
            continue
        pattern = condition.split(", from ")[0]
        if not (pattern.startswith("? ") and pattern.endswith(" ?")):
            continue
        if q in pool:
            assert pool[q] == row["input"][0], f"同题多种QA输入: {q}"
        pool[q] = row["input"][0]

rng = random.Random(20261105)
entity_pool = sorted(q for q in pool if gold[q][1] == "entity")
time_pool = sorted(q for q in pool if gold[q][1] == "time")
assert len(entity_pool) >= 200
selected = rng.sample(entity_pool, 200) + time_pool
rng.shuffle(selected)

jobs = []
for q in selected:
    prompt = pool[q]
    match = list(qa_re.finditer(prompt))[-1]
    body, interval = match[2].rsplit(", from ", 1)
    start, end = interval.split(" to ", 1)
    jobs.append(dict(
        question=q, answer=gold[q][0], answer_type=gold[q][1],
        A_prompt=prompt, facts=ast.literal_eval(match[3]),
        relation=body[2:-2], time=[start, end],
    ))

def save(name, value):
    (out / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

save("settings.json", dict(
    seed=20261105, source=str(source),
    entity_count=200, time_count=len(time_pool), excluded=sorted(excluded)
))
(out / "gp-B.txt").write_text(gp_text, encoding="utf-8")
save("jobs.json", jobs)
print(f"选中: 实体200题；双通配符时间题{len(time_pool)}题", flush=True)

async def run():
    clients = {
        arm: AsyncLLMClient(
            api_key="dummy", base_url="http://127.0.0.1:8000/v1",
            model_name="qwen14b", cache_path=str(out / f"{arm}.jsonl")
        )
        for arm in ("GP", "A", "B")
    }
    semaphore = asyncio.Semaphore(8)

    async def repair(job):
        relation, interval = job["relation"], job["time"]
        if job["answer_type"] == "entity" and relation == "?":
            return dict(
                accepted=False, request_failed=False, attempted=False,
                response="Skipped: relation is also unknown",
                fact_pattern=["?", relation, "?"]
            )

        prompt = gp_text.format(
            question=job["question"], facts=job["facts"]
        )
        answer_rule = (
            "Keep the requested answer entity as ?. Do not guess an answer.\n"
            if job["answer_type"] == "entity" else
            "This is a time question: retain the entities specified in the question.\n"
        )
        prompt += (
            "\nGraph index compatibility correction:\n"
            f"The previous triple was: ?; {relation}; ?\n"
            "The graph index supports at most ONE ? in the three fact slots.\n"
            "Re-read the question and retain the known participant in its correct slot.\n"
            + answer_rule +
            "A temporal reference entity does not replace the participant being queried.\n"
            f"Keep the relation EXACTLY: {relation}\n"
            f"Keep the time EXACTLY: <time> {interval[0]}; {interval[1]} </time>\n"
            "Output only <fact> and <time> lines. If no known participant can be\n"
            "identified from the question and facts, retain the previous triple.\n"
        )
        async with semaphore:
            response, _ = await clients["GP"].request(
                prompt=prompt, temperature=0,
                max_tokens=4096, use_cache=True
            )
        failed = (
            prompt, "qwen14b", 4096, None, False
        ) not in clients["GP"].cache
        fm = re.search(r"<fact>(.*?)</fact>", response, re.S)
        tm = re.search(r"<time>(.*?)</time>", response, re.S)
        fp = [x.strip() for x in fm[1].split(";")] if fm else []
        tp = [x.strip() for x in tm[1].split(";")] if tm else []
        accepted = (
            not failed and len(fp) == 3 and all(fp)
            and fp.count("?") <= 1
            and fp[1] == relation and tp == interval
        )
        if accepted and job["answer_type"] == "entity":
            accepted = (fp[0] == "?") != (fp[2] == "?")

        return dict(
            accepted=accepted, request_failed=failed, attempted=True,
            response=response,
            fact_pattern=fp if accepted else ["?", relation, "?"]
        )

    async def qa(arm, i):
        client, job = clients[arm], jobs[i]
        prompt = job[f"{arm}_prompt"]
        async with semaphore:
            prediction, _ = await get_qa_result(prompt, client)
        key = (prompt, "qwen14b", 4096, None, False)
        raw = client.cache.get(key, "")
        failed = key not in client.cache
        return arm, i, dict(
            prediction=prediction, response=raw,
            request_failed=failed,
            missing_marker=not failed and "So the answer is" not in raw,
            correct=not failed and is_correct(job["answer"], prediction)
        )

    try:
        models = await clients["GP"].async_client.models.list()
        assert "qwen14b" in {m.id for m in models.data}, "Qwen服务未就绪"

        repairs = await tqdm_asyncio.gather(
            *(repair(j) for j in jobs),
            desc="纠正GP", total=len(jobs)
        )

        graph = None
        if any(r["accepted"] for r in repairs):
            cache = Path("artifacts/multitq_graph_retriever.pkl")
            assert cache.is_file(), f"图缓存缺失: {cache}"
            graph = build_graph_retriever(SimpleNamespace(
                graph_cache_path=str(cache), force_reload=False,
                text_retriever_model_name=str(Path("models/bge-m3").resolve()),
                device="cuda:0"
            ))

        for job, result in zip(jobs, repairs):
            result["graph_count"] = 0
            job["B_prompt"] = job["A_prompt"]

            if result["accepted"]:
                graph_facts = graph.get_facts(
                    tuple(result["fact_pattern"]), tuple(job["time"])
                )
                result["graph_count"] = len(graph_facts)

                # 查空时完整保留A输入，不只修改过滤条件。
                if graph_facts:
                    text = graph.facts_to_text(graph_facts)
                    facts = (
                        uniform_sample_preserve_edges(text, sample_size=50)
                        if len(text) >= 50 else text
                    )
                    condition = (
                        " ".join(result["fact_pattern"])
                        + f", from {job['time'][0]} to {job['time'][1]}"
                    )
                    match = list(qa_re.finditer(job["A_prompt"]))[-1]
                    prompt = job["A_prompt"]
                    prompt = (
                        prompt[:match.start(3)]
                        + str(facts) + prompt[match.end(3):]
                    )
                    job["B_prompt"] = (
                        prompt[:match.start(2)]
                        + condition + prompt[match.end(2):]
                    )

            result["qa_input_changed"] = job["B_prompt"] != job["A_prompt"]

        save("jobs.json", jobs)
        save("repairs.json", repairs)

        tasks = [("A", i) for i in range(len(jobs))]
        tasks += [
            ("B", i) for i, r in enumerate(repairs)
            if r["qa_input_changed"]
        ]
        rng.shuffle(tasks)
        results = await tqdm_asyncio.gather(
            *(qa(arm, i) for arm, i in tasks),
            desc="QA A/B", total=len(tasks)
        )
    finally:
        await asyncio.gather(
            *(c.async_client.close() for c in clients.values())
        )

    indexed = {(arm, i): value for arm, i, value in results}
    pairs = []
    for i, (job, result) in enumerate(zip(jobs, repairs)):
        a = indexed["A", i]
        b = indexed["B", i] if result["qa_input_changed"] else a
        pairs.append(dict(
            question=job["question"], answer=job["answer"],
            answer_type=job["answer_type"], repair=result,
            B_reused_A=not result["qa_input_changed"], A=a, B=b
        ))
    save("comparison.json", pairs)

    failures = {
        arm: sum(
            v["request_failed"]
            for (a, i), v in indexed.items() if a == arm
        )
        for arm in ("A", "B")
    }
    missing = {
        arm: sum(
            v["missing_marker"]
            for (a, i), v in indexed.items() if a == arm
        )
        for arm in ("A", "B")
    }
    contract = dict(
        attempted=sum(r["attempted"] for r in repairs),
        request_failed=sum(r["request_failed"] for r in repairs),
        accepted=sum(r["accepted"] for r in repairs),
        graph_nonempty=sum(r["graph_count"] > 0 for r in repairs),
        qa_input_changed=sum(r["qa_input_changed"] for r in repairs),
    )
    print("GP纠正统计:", contract)
    print("QA请求失败数:", failures)
    print("缺少答案标记:", missing)
    print("B输入未变、复用A回答:", len(jobs) - contract["qa_input_changed"])

    summary = {}
    if not any(failures.values()) and not contract["request_failed"]:
        for kind in ("entity", "time"):
            rows = [p for p in pairs if p["answer_type"] == kind]
            if not rows:
                continue
            a = sum(p["A"]["correct"] for p in rows)
            b = sum(p["B"]["correct"] for p in rows)
            gained = sum(
                not p["A"]["correct"] and p["B"]["correct"] for p in rows
            )
            lost = sum(
                p["A"]["correct"] and not p["B"]["correct"] for p in rows
            )
            summary[kind] = dict(
                total=len(rows), A=a, B=b, gained=gained, lost=lost
            )
            print(
                f"{kind}: A={a}/{len(rows)}, B={b}/{len(rows)}, "
                f"A错B对={gained}, A对B错={lost}, 净增={b-a:+d}"
            )
    else:
        print("存在请求失败，先排查失败，不作优劣结论。")

    save("summary.json", dict(
        GP=contract, QA_failures=failures,
        missing_marker=missing, by_type=summary
    ))

asyncio.run(run())
print("结果目录:", out)
