import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from tqdm.asyncio import tqdm_asyncio
from utils.client import AsyncLLMClient
from utils.eval_utils import is_correct, parse_list_str
from utils.io_utils import load_qas
from utils.query_process import get_qa_result

out = Path(os.environ["LABEL_HOLDOUT_RUN"])
source = Path("artifacts/gp-B-full-MqkyPV/llm_cache.jsonl")
rule_path = Path("artifacts/qa-entity-label-G85HfM/extra-rule.txt")
assert source.is_file(), f"找不到全量 B 缓存: {source}"
rule = rule_path.read_text(encoding="utf-8")
assert rule.strip(), "实体标签规则为空"
(out / "extra-rule.txt").write_text(rule, encoding="utf-8")

excluded = set()

def collect_questions(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"question", "Question", "题目"} and isinstance(item, str):
                excluded.add(item)
            else:
                collect_questions(item)
    elif isinstance(value, list):
        for item in value:
            collect_questions(item)

previous = [
    "artifacts/gp-wildcard-ab/questions.json",
    "artifacts/gp-holdout-2eEr22/questions.json",
    "artifacts/gp-role-ab-Paj2eX/questions.json",
    "artifacts/qa-order-ab-34VEXw/jobs.json",
    "artifacts/manual-pattern-diag-2M9xVw/report.json",
    "artifacts/qa-entity-label-G85HfM/cases.json",
]
for name in previous:
    collect_questions(json.loads(Path(name).read_text(encoding="utf-8")))

excluded.add(
    "Before China, which country was the last to receive an appeal "
    "from Thailand to the Office of the Attorney General?"
)

qs, answers, _, kinds = load_qas("datasets/MultiTQ/questions/test.json")
frequency = Counter(qs)

# 排除最初 seed=42 的 200 题试跑。
excluded.update(qs[i] for i in random.Random(42).sample(range(len(qs)), 200))

rng = random.Random(20261104)
selected = []
for kind in ("entity", "time"):
    pool = [
        i for i, q in enumerate(qs)
        if kinds[i] == kind and frequency[q] == 1 and q not in excluded
    ]
    assert len(pool) >= 100, f"{kind} 剩余题数不足"
    selected.extend(rng.sample(pool, 100))

rng.shuffle(selected)
wanted = {qs[i] for i in selected}

print("扫描 B 全量缓存，找回 200 个原始 QA 输入……", flush=True)
qa_re = re.compile(r"(?m)^Question: ([^\n]*)\nFacts filter condition:")
prompts = {}

with source.open(encoding="utf-8") as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["input"]
        if key[1] != "qwen14b":
            continue
        found = qa_re.findall(key[0])
        if not found or found[-1] not in wanted:
            continue
        q, prompt = found[-1], key[0]
        if q in prompts:
            assert prompts[q] == prompt, f"同题存在不同 QA 输入: {q}"
        prompts[q] = prompt

assert set(prompts) == wanted, f"QA 输入缺失: {sorted(wanted - set(prompts))}"

jobs = [
    dict(
        question=qs[i],
        answer=answers[i],
        answer_type=kinds[i],
        prompt=prompts[qs[i]],
    )
    for i in selected
]

def save(name, value):
    (out / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

save("jobs.json", jobs)
save("settings.json", dict(
    seed=20261104,
    model="qwen14b",
    temperature=0,
    max_tokens=4096,
    concurrency=8,
    source=str(source),
    rule_source=str(rule_path),
    excluded_questions=sorted(excluded),
))
print("原始输入覆盖: 200/200；实体 100 题，时间 100 题", flush=True)

async def run():
    clients = {
        arm: AsyncLLMClient(
            api_key="dummy",
            base_url="http://127.0.0.1:8000/v1",
            model_name="qwen14b",
            cache_path=str(out / f"{arm}.jsonl"),
        )
        for arm in ("A", "B")
    }
    semaphore = asyncio.Semaphore(8)

    async def worker(arm, i):
        job, client = jobs[i], clients[arm]
        prompt = job["prompt"] + (rule if arm == "B" else "")
        async with semaphore:
            prediction, _ = await get_qa_result(prompt, client)

        key = (prompt, "qwen14b", 4096, None, False)
        failed = key not in client.cache
        raw = client.cache.get(key, "")
        return arm, i, dict(
            prediction=prediction,
            scored_value=parse_list_str(prediction),
            correct=not failed and is_correct(job["answer"], prediction),
            request_failed=failed,
            missing_marker=not failed and "So the answer is" not in raw,
            response=raw,
        )

    # 随机交错两组请求，共享并发限制。
    tasks = [(arm, i) for i in range(len(jobs)) for arm in ("A", "B")]
    rng.shuffle(tasks)

    try:
        models = await clients["A"].async_client.models.list()
        assert "qwen14b" in {m.id for m in models.data}, "未发现 qwen14b 服务"
        results = await tqdm_asyncio.gather(
            *(worker(arm, i) for arm, i in tasks),
            desc="QA A/B",
            total=400,
        )
    finally:
        await asyncio.gather(
            *(c.async_client.close() for c in clients.values())
        )

    indexed = {(arm, i): value for arm, i, value in results}
    paired = [
        dict(
            question=job["question"],
            answer=job["answer"],
            answer_type=job["answer_type"],
            A=indexed["A", i],
            B=indexed["B", i],
        )
        for i, job in enumerate(jobs)
    ]
    failed = {
        arm: sum(p[arm]["request_failed"] for p in paired)
        for arm in clients
    }
    missing = {
        arm: sum(p[arm]["missing_marker"] for p in paired)
        for arm in clients
    }
    save("comparison.json", paired)

    print("请求失败数:", failed)
    print("缺少答案标记:", missing)

    if any(failed.values()):
        print("存在失败请求，本轮不作 A/B 优劣结论。逐题结果已保存。")
        return

    summary = {}
    for kind in ("entity", "time"):
        rows = [p for p in paired if p["answer_type"] == kind]
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
            f"{kind}: A={a}/100, B={b}/100, A错B对={gained}, "
            f"A对B错={lost}, 净增={b-a:+d}"
        )

    save("summary.json", dict(
        request_failed=failed,
        missing_marker=missing,
        by_type=summary,
    ))

asyncio.run(run())
print("结果目录:", out)
