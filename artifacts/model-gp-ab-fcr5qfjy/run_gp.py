import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from tqdm.asyncio import tqdm_asyncio
from utils.client import AsyncLLMClient

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--expected-root", required=True)
parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

assert re.fullmatch(r"[A-Za-z0-9_-]+", args.tag)
root = Path(args.run_dir).resolve()
out = root / args.tag
expected_root = Path(args.expected_root).expanduser().resolve()
jobs_bytes = (root / "jobs.json").read_bytes()
jobs = json.loads(jobs_bytes)

assert len(jobs) == 200
assert Counter(j["answer_type"] for j in jobs) == {
    "entity": 100, "time": 100
}
assert len({j["question"] for j in jobs}) == 200

for job in jobs:
    assert hashlib.sha256(
        job["prompt"].encode("utf-8")
    ).hexdigest() == job["prompt_sha256"], job["question"]

def json_hash(value):
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temporary.replace(path)

common_path = root / "sampling-defaults/generation_config.json"
common = json.loads(common_path.read_text(encoding="utf-8"))
preflight = json.loads(
    (root / "preflight.json").read_text(encoding="utf-8")
)
assert json_hash(common) == json_hash(
    preflight["generation_config.json"]
), "冻结生成配置已变化"

config = json.loads(
    (expected_root / "config.json").read_text(encoding="utf-8")
)
if args.model == "qwen14b":
    original = json.loads(
        (expected_root / "generation_config.json")
        .read_text(encoding="utf-8")
    )
    assert json_hash(original) == json_hash(common), "14B生成配置已变化"

metadata = dict(
    model=args.model,
    expected_root=str(expected_root),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    client_sha256=hashlib.sha256(
        Path("src/utils/client.py").read_bytes()
    ).hexdigest(),
    jobs_sha256=hashlib.sha256(jobs_bytes).hexdigest(),
    model_config_sha256=json_hash(config),
    common_generation_sha256=json_hash(common),
    common_generation=common,
    base_url=args.base_url,
    temperature=0,
    max_tokens=4096,
    concurrency=8,
    max_format_attempts=10,
)

completed = {}
if out.exists():
    assert args.resume, f"结果目录已存在: {out}"
    assert json.loads(
        (out / "metadata.json").read_text(encoding="utf-8")
    ) == metadata, "续跑配置与原记录不同"
    done_path = out / "results.jsonl"
    if done_path.exists():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row["request_failed"]:
                    completed[row["question"]] = row
    assert set(completed) <= {j["question"] for j in jobs}
    for job in jobs:
        if job["question"] in completed:
            assert (
                completed[job["question"]]["prompt_sha256"]
                == job["prompt_sha256"]
            )

async def run():
    client = AsyncLLMClient(
        api_key="dummy", base_url=args.base_url,
        model_name=args.model,
        cache_path=str(out / "cache.jsonl")
    )
    semaphore = asyncio.Semaphore(8)

    async def worker(job):
        if job["question"] in completed:
            return completed[job["question"]]

        attempts = []
        fact, interval = ("?", "?", "?"), ("?", "?")
        parsed = failed = False

        async with semaphore:
            for number in range(10):
                response, ok = await client.request(
                    prompt=job["prompt"],
                    temperature=0,
                    max_tokens=4096,
                    use_cache=(number == 0),
                )
                attempts.append(dict(
                    number=number + 1,
                    request_ok=bool(ok),
                    response=response
                ))
                if not ok:
                    failed = True
                    break

                fm = re.search(
                    r"<fact>(.*?)</fact>", response, re.DOTALL
                )
                tm = re.search(
                    r"<time>(.*?)</time>", response, re.DOTALL
                )
                if fm is None or tm is None:
                    continue

                fp = [
                    x.strip()
                    for x in fm.group(1).strip().split(";")
                ]
                tp = [
                    x.strip()
                    for x in tm.group(1).strip().split(";")
                ]
                if len(fp) != 3 or len(tp) != 2:
                    continue

                fact, interval, parsed = tuple(fp), tuple(tp), True
                break

        row = dict(
            question=job["question"],
            answer_type=job["answer_type"],
            prompt_sha256=job["prompt_sha256"],
            fact_pattern=fact,
            time_pattern=interval,
            parsed=parsed,
            request_failed=failed,
            format_failure=not parsed and not failed,
            request_attempts=len(attempts),
            index_shape_supported=parsed and fact.count("?") <= 1,
            double_entity_wildcard=(
                parsed and fact[0] == "?" and fact[2] == "?"
            ),
            attempts=attempts,
        )
        with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    try:
        models = await client.async_client.models.list()
        served = [m for m in models.data if m.id == args.model]
        assert len(served) == 1, f"未发现模型: {args.model}"

        actual_root = getattr(served[0], "root", None)
        assert (
            actual_root
            and Path(actual_root).expanduser().resolve() == expected_root
        ), f"接口实际模型路径不符: {actual_root}"

        if not out.exists():
            out.mkdir()
            save(out / "metadata.json", metadata)

        save(out / "runtime.json", dict(
            api_model=args.model,
            api_root=actual_root,
            max_model_len=getattr(served[0], "max_model_len", None),
            common_generation_sha256=json_hash(common),
        ))
        results = await tqdm_asyncio.gather(
            *(worker(j) for j in jobs),
            desc=args.tag, total=len(jobs)
        )
    finally:
        await client.async_client.close()

    save(out / "results.json", results)
    summary = {}
    for kind in ("entity", "time"):
        rows = [r for r in results if r["answer_type"] == kind]
        summary[kind] = dict(
            total=len(rows),
            request_failed=sum(r["request_failed"] for r in rows),
            format_failure=sum(r["format_failure"] for r in rows),
            parsed=sum(r["parsed"] for r in rows),
            index_shape_supported=sum(
                r["index_shape_supported"] for r in rows
            ),
            double_entity_wildcard=sum(
                r["double_entity_wildcard"] for r in rows
            ),
            request_attempts=sum(r["request_attempts"] for r in rows),
        )

    save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("以上是格式与模式形状统计，不是语义正确率或QA准确率。")
    print("结果目录:", out)

asyncio.run(run())
