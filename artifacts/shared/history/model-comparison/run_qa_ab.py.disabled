import asyncio
import ast
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("src").resolve()))
from tqdm.asyncio import tqdm_asyncio
from run_pipeline import build_graph_retriever
from utils.client import AsyncLLMClient
from utils.eval_utils import is_correct
from utils.io_utils import read_prompt
from utils.query_process import get_qa_result
from utils.sample_utils import uniform_sample_preserve_edges

root = Path("artifacts/model-gp-ab-fcr5qfjy").resolve()
jobs_bytes = (root / "jobs.json").read_bytes()
jobs = json.loads(jobs_bytes)
assert len(jobs) == 200
assert Counter(j["answer_type"] for j in jobs) == {"entity": 100, "time": 100}
assert len({j["question"] for j in jobs}) == 200

metas = {
    arm: json.loads((root / tag / "metadata.json").read_text(encoding="utf-8"))
    for arm, tag in (("A", "GP14"), ("B", "GP32"))
}
assert metas["A"]["model"] == "qwen14b"
assert metas["B"]["model"] == "qwen32b-gp"

for key in (
    "runner_sha256", "client_sha256", "jobs_sha256",
    "common_generation_sha256", "temperature",
    "max_tokens", "concurrency", "max_format_attempts"
):
    assert metas["A"][key] == metas["B"][key], f"GP对照条件不同: {key}"

assert hashlib.sha256(jobs_bytes).hexdigest() == metas["A"]["jobs_sha256"]
assert hashlib.sha256(
    Path("src/utils/client.py").read_bytes()
).hexdigest() == metas["A"]["client_sha256"]

common = json.loads(
    (root / "sampling-defaults/generation_config.json")
    .read_text(encoding="utf-8")
)
assert hashlib.sha256(json.dumps(
    common, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode()).hexdigest() == metas["A"]["common_generation_sha256"]

gp = {}
for arm, tag in (("A", "GP14"), ("B", "GP32")):
    rows = json.loads(
        (root / tag / "results.json").read_text(encoding="utf-8")
    )
    assert len(rows) == len(jobs)
    assert len({r["question"] for r in rows}) == len(jobs)
    gp[arm] = {r["question"]: r for r in rows}
    assert set(gp[arm]) == {j["question"] for j in jobs}
    for j in jobs:
        r = gp[arm][j["question"]]
        assert r["prompt_sha256"] == j["prompt_sha256"]
        assert r["parsed"] and not r["request_failed"] and not r["format_failure"]
        assert len(r["fact_pattern"]) == 3 and len(r["time_pattern"]) == 2

# 恢复原始文本检索事实，供图检索为空时回退。
gp_template = read_prompt(str(root / "gp-B.txt"))
before, after = gp_template.split("{facts}", 1)
fallback = {}

for job in jobs:
    prompt = job["prompt"]
    assert hashlib.sha256(prompt.encode()).hexdigest() == job["prompt_sha256"]
    prefix = before.format(question=job["question"])
    assert prompt.startswith(prefix) and prompt.endswith(after)
    text = (
        prompt[len(prefix):len(prompt)-len(after)]
        if after else prompt[len(prefix):]
    )
    facts = ast.literal_eval(text)
    assert isinstance(facts, list) and all(isinstance(f, str) for f in facts)
    fallback[job["question"]] = facts[:50]

qa_source = Path("artifacts/gp-B-full-MqkyPV/qa.txt")
qa_template = read_prompt(str(qa_source))

for name in (
    "artifacts/multitq_graph_retriever.pkl",
    "MultiTQ_entity_embeddings.npy",
    "MultiTQ_relation_embeddings.npy"
):
    assert Path(name).is_file(), f"缺少已有索引文件，停止而不重新建库: {name}"

out = Path(tempfile.mkdtemp(prefix="qa-ab-", dir=str(root)))

def save(name, value):
    (out / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

save("settings.json", dict(
    A="14B GP + 14B QA",
    B="32B GP + 14B QA",
    jobs_sha256=metas["A"]["jobs_sha256"],
    common_generation_sha256=metas["A"]["common_generation_sha256"],
    qa_source=str(qa_source), qa_model="qwen14b",
    temperature=0, max_tokens=4096, concurrency=8,
    source_sha256={
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "src/utils/client.py", "src/utils/query_process.py",
            "src/utils/sample_utils.py",
            "src/retriever/graph_retriever.py",
            "src/retriever/text_retriever.py"
        )
    },
    gp_result_sha256={
        tag: hashlib.sha256((root / tag / "results.json").read_bytes()).hexdigest()
        for tag in ("GP14", "GP32")
    },
))
(out / "qa.txt").write_text(qa_template, encoding="utf-8")
print("本次目录:", out, flush=True)

async def run():
    clients = {
        arm: AsyncLLMClient(
            api_key="dummy", base_url="http://127.0.0.1:8000/v1",
            model_name="qwen14b", cache_path=str(out / f"{arm}.jsonl")
        )
        for arm in ("A", "B")
    }
    semaphore = asyncio.Semaphore(8)
    prepared = []

    async def worker(arm, i):
        item, client = prepared[i], clients[arm]
        prompt = item[arm]["prompt"]
        async with semaphore:
            prediction, _ = await get_qa_result(prompt, client)
        key = (prompt, "qwen14b", 4096, None, False)
        failed = key not in client.cache
        response = client.cache.get(key, "")
        return arm, i, dict(
            prediction=prediction, response=response,
            request_failed=failed,
            missing_marker=not failed and "So the answer is" not in response,
            correct=not failed and is_correct(item["answer"], prediction)
        )

    try:
        models = await clients["A"].async_client.models.list()
        served = [m for m in models.data if m.id == "qwen14b"]
        assert len(served) == 1, "请先恢复14B服务"
        actual_root = getattr(served[0], "root", None)
        assert (
            actual_root
            and Path(actual_root).resolve()
            == Path("models/qwen14b-awq").resolve()
        )
        save("runtime.json", dict(
            qa_model="qwen14b", api_root=actual_root,
            common_generation_sha256=metas["A"]["common_generation_sha256"]
        ))

        graph = build_graph_retriever(SimpleNamespace(
            graph_cache_path="artifacts/multitq_graph_retriever.pkl",
            force_reload=False,
            text_retriever_model_name=str(Path("models/bge-m3").resolve()),
            device="cuda:0"
        ))

        for job in jobs:
            q = job["question"]
            item = dict(
                question=q, answer=job["answer"],
                answer_type=job["answer_type"]
            )
            for arm in ("A", "B"):
                result = gp[arm][q]
                fp = tuple(result["fact_pattern"])
                tp = tuple(result["time_pattern"])
                found = graph.get_facts(fp, tp)
                texts = graph.facts_to_text(found)

                if len(texts) == 0:
                    facts = fallback[q]
                elif len(texts) >= 50:
                    facts = uniform_sample_preserve_edges(texts, sample_size=50)
                else:
                    facts = texts

                # 保留作者流程：过滤条件使用原始GP名称和时间。
                condition = " ".join(fp) + f", from {tp[0]} to {tp[1]}"
                prompt = qa_template.format(
                    question=q, facts=facts, filter=condition
                )
                item[arm] = dict(
                    fact_pattern=fp, time_pattern=tp,
                    graph_count=len(found), facts=facts,
                    qa_count=len(facts), used_text_fallback=not texts,
                    filter=condition, prompt=prompt
                )

            item["shared_input"] = item["A"]["prompt"] == item["B"]["prompt"]
            prepared.append(item)

        save("jobs.json", prepared)
        tasks = [("A", i) for i in range(len(prepared))]
        tasks += [
            ("B", i) for i, item in enumerate(prepared)
            if not item["shared_input"]
        ]
        random.Random(1432).shuffle(tasks)
        print("同一14B执行QA，实际请求数:", len(tasks), flush=True)
        results = await tqdm_asyncio.gather(
            *(worker(arm, i) for arm, i in tasks),
            desc="QA A/B", total=len(tasks)
        )
    finally:
        await asyncio.gather(
            *(c.async_client.close() for c in clients.values())
        )

    indexed = {(arm, i): value for arm, i, value in results}
    paired = []
    for i, item in enumerate(prepared):
        a = indexed["A", i]
        b = a if item["shared_input"] else indexed["B", i]
        paired.append(dict(
            question=item["question"], answer=item["answer"],
            answer_type=item["answer_type"],
            shared_input=item["shared_input"], A=a, B=b
        ))
    save("comparison.json", paired)

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
    print("QA请求失败数:", failures)
    print("缺少答案标记:", missing)
    print(
        "输入相同、共享回答的题数:",
        sum(p["shared_input"] for p in paired)
    )

    summary = {}
    for kind in ("entity", "time"):
        indices = [
            i for i, j in enumerate(prepared)
            if j["answer_type"] == kind
        ]
        counts = {
            arm: sum(prepared[i][arm]["used_text_fallback"] for i in indices)
            for arm in ("A", "B")
        }
        print(f"{kind} 图检索为空、回退文本的题数:", counts)
        if any(failures.values()):
            continue

        rows = [paired[i] for i in indices]
        a, b = (
            sum(p[arm]["correct"] for p in rows) for arm in ("A", "B")
        )
        gained = sum(
            not p["A"]["correct"] and p["B"]["correct"] for p in rows
        )
        lost = sum(
            p["A"]["correct"] and not p["B"]["correct"] for p in rows
        )
        summary[kind] = dict(
            total=len(rows), A=a, B=b,
            gained=gained, lost=lost, fallback=counts
        )
        print(
            f"{kind}: A={a}/{len(rows)}, B={b}/{len(rows)}, "
            f"A错B对={gained}, A对B错={lost}, 净增={b-a:+d}"
        )

    if any(failures.values()):
        print("存在失败请求，本轮先不作优劣结论。")

    save("summary.json", dict(
        A="14B GP + 14B QA", B="32B GP + 14B QA",
        request_failed=failures, missing_marker=missing,
        shared_input=sum(p["shared_input"] for p in paired),
        by_type=summary
    ))

asyncio.run(run())
print("结果目录:", out)
