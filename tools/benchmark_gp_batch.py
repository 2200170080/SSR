"""Compare GP throughput at batch-token budgets 2048 and 4096; restore 2048."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path.cwd().resolve()
PILOT = PROJECT / "artifacts/model-gp-ab-fcr5qfjy"
POINTER = PILOT / "gp-batch-bench-path.txt"
sys.path.insert(0, str(PROJECT / "src"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def get_json(route):
    with urllib.request.urlopen("http://127.0.0.1:8000" + route, timeout=10) as response:
        return json.load(response)


def full_module(full):
    manifest = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
    require(digest(full / "run_full.py") == manifest["runner_sha256"], "冻结全量脚本校验失败")
    spec = importlib.util.spec_from_file_location("ssr_frozen_full_for_benchmark", full / "run_full.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(full)
    return module, manifest


def validate_directory(folder, prefix):
    require(folder.parent == (PROJECT / "artifacts").resolve() and folder.name.startswith(prefix), "实验目录不在预期位置")


def service_state(expected_budget=None):
    saved = json.loads((PILOT / "service-32.json").read_text(encoding="utf-8"))
    command = saved["command"]
    def option(name, default=None):
        return command[command.index(name) + 1] if name in command else default
    budget = int(option("--max-num-batched-tokens", "2048"))
    require(saved["model"] == "qwen32b-gp" and Path(saved["root"]).resolve() == (PROJECT / "models/qwen32b-awq").resolve(), "记录的服务不是本项目32B")
    require(option("--max-num-seqs") == "8" and option("--max-model-len") == "8192", "服务并发或上下文配置已变化")
    require(option("--gpu-memory-utilization") == "0.80" and option("--dtype") == "half", "服务显存或计算精度配置已变化")
    require(Path(option("--generation-config")).resolve() == (PILOT / "sampling-defaults").resolve(), "生成默认配置路径已变化")
    if expected_budget is not None:
        require(budget == expected_budget, f"服务批处理预算应为{expected_budget}，实际为{budget}")
    models = get_json("/v1/models")["data"]
    require(any(m["id"] == saved["model"] and Path(m["root"]).resolve() == Path(saved["root"]).resolve() for m in models), "接口实际模型与服务记录不符")
    proc = Path("/proc") / str(saved["pid"])
    require(proc.stat().st_uid == os.getuid(), "服务进程不属于当前用户")
    fields = (proc / "stat").read_text().rpartition(")")[2].split()
    require(fields[0] != "Z" and fields[19] == saved["start"], "服务PID身份已变化")
    return saved


def derived_switch(source, budget):
    ready = "if served and any(\n"
    seqs = '    "--max-num-seqs", str(target["seqs"]),\n'
    require(source.count(ready) == source.count(seqs) == 1, "服务脚本结构已变化，停止生成测速启动器")
    # A fresh server in EACH arm prevents previous prefix-cache contents from biasing timing.
    result = source.replace(ready, "if False and served and any(\n", 1)
    return result.replace(seqs, seqs + f'    "--max-num-batched-tokens", "{budget}",\n', 1)


def prepare():
    full = Path((PILOT / "full-32b-path.txt").read_text(encoding="utf-8").strip()).resolve()
    validate_directory(full, "qwen32b-full-")
    module, source_manifest = full_module(full)
    require(source_manifest["models"] == {"gp": "qwen32b-gp", "qa": "qwen32b-gp"}, "全量配置不是32B")
    original = service_state(2048)
    jobs, warmup, stats = choose_jobs(full / "bulk/inputs.jsonl")
    require(stats["input_sha256"] == source_manifest["input_sha256"], "原始输入校验不一致")
    # Shuffle once, identically for both arms; selection uses answer TYPE, never answer correctness.
    random.Random(20261104).shuffle(jobs)
    random.Random(20261105).shuffle(warmup)
    template = (full / "gp.txt").read_text(encoding="utf-8")
    for job in jobs + warmup:
        question, _ = module.parse_gp_prompt(job["prompt"], template)
        require(question == job["question"], "抽样请求与冻结GP模板不匹配")
    source = (PILOT / "switch_model.py").read_text(encoding="utf-8")
    switchers = {n: derived_switch(source, n) for n in (2048, 4096)}
    folder = Path(tempfile.mkdtemp(prefix="gp-batch-bench-", dir=PROJECT / "artifacts"))
    (folder / ".gitignore").write_text("bulk/\n*.tmp\n*.lock\n", encoding="utf-8")
    (folder / "bulk").mkdir()
    write_json(folder / "bulk/jobs.json", jobs)
    write_json(folder / "bulk/warmup.json", warmup)
    (folder / "gp.txt").write_text(template, encoding="utf-8", newline="\n")
    for budget, text in switchers.items():
        (folder / f"switch-{budget}.py").write_text(text, encoding="utf-8", newline="\n")
    meta = dict(version="SSR-GP-batch-throughput-v1", full_run=str(full), source_signature=source_manifest["signature"],
        created=datetime.now(timezone.utc).isoformat(), runner_sha256=digest(__file__),
        original_service=original, original_switch_sha256=digest(PILOT / "switch_model.py"),
        sampling=stats, budgets=[2048, 4096], concurrency=8, temperature=0, max_tokens=4096,
        jobs_sha256=digest(folder / "bulk/jobs.json"), warmup_sha256=digest(folder / "bulk/warmup.json"),
        template_sha256=digest(folder / "gp.txt"),
        switch_sha256={str(n): digest(folder / f"switch-{n}.py") for n in (2048, 4096)})
    write_json(folder / "manifest.json", meta)
    POINTER.write_text(str(folder) + "\n", encoding="utf-8")
    print("测速输入准备完成：200题（实体100，时间100），另有8题预热。", flush=True)
    print("本次测速目录:", folder, flush=True)
    print("当前全量任务尚未停止。先在其tmux中按Ctrl+C，等脚本退出后执行run。", flush=True)


def validate_benchmark(folder):
    validate_directory(folder, "gp-batch-bench-")
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    require(digest(__file__) == meta["runner_sha256"], "测速脚本已变化")
    require(digest(PILOT / "switch_model.py") == meta["original_switch_sha256"], "原服务脚本已变化")
    for relative, key in (("bulk/jobs.json", "jobs_sha256"), ("bulk/warmup.json", "warmup_sha256"), ("gp.txt", "template_sha256")):
        require(digest(folder / relative) == meta[key], "测速输入已变化: " + relative)
    for budget in (2048, 4096):
        require(digest(folder / f"switch-{budget}.py") == meta["switch_sha256"][str(budget)], "测速服务脚本已变化")
    full = Path(meta["full_run"])
    module, source_manifest = full_module(full)
    require(source_manifest["signature"] == meta["source_signature"], "全量实验版本已变化")
    return meta, full, module


def guarded_switch(folder, budget, restoring=False):
    pending = []
    def defer(signum, frame):
        if not pending:
            print("收到中断；等当前服务切换退出后，恢复2048配置。", flush=True)
        pending.append(signum)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, defer)
    try:
        # The controller, rather than Ctrl+C, owns the helper lifetime and restoration ordering.
        child = subprocess.Popen([sys.executable, str(folder / f"switch-{budget}.py"), "32"], start_new_session=True)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        require(code == 0, f"切换到{budget}失败，退出码{code}")
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if pending and not restoring:
        raise KeyboardInterrupt("用户中断测速")


def restore_service(folder):
    def defer(signum, frame):
        print("正在恢复2048服务，请等待恢复检查结束。", flush=True)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, defer)
    try:
        guarded_switch(folder, 2048, restoring=True)
        write_json(folder / "restore-status.json", dict(restored=True, budget=2048, service=service_state(2048)))
        print("2048服务已恢复。可使用原全量目录的resume.sh继续。", flush=True)
    except BaseException as error:
        write_json(folder / "restore-status.json", dict(restored=False, error=repr(error)))
        print("恢复失败；先执行本脚本restore，查看服务日志，不要重新创建全量目录。", flush=True)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def snapshot(folder, name):
    bulk = folder / "bulk"
    try:
        command = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu", "--format=csv"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        (bulk / f"{name}-gpu.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as response:
            (bulk / f"{name}-metrics.txt").write_bytes(response.read())
    except Exception as error:
        print("辅助监控未采集:", type(error).__name__, str(error), flush=True)


def compare(folder, arms, patterns):
    a, b = arms[2048], arms[4096]
    same = sum(patterns[2048].get(k) == patterns[4096].get(k) for k in patterns[2048] if k in patterns[4096])
    speedup = b["jobs_per_second"] / a["jobs_per_second"] if a["jobs_per_second"] > 0 else None
    clean = all(s.get("completed") == s.get("success") == 200 and s.get("comparison_eligible", False)
        and s.get("usage_complete", False) for s in arms.values())
    result = dict(arms={str(k): v for k, v in arms.items()}, speedup_4096_over_2048=speedup,
        matching_patterns=same, compared_patterns=len(set(patterns[2048]) & set(patterns[4096])),
        clean_comparison=clean, notice="图模式一致率不是QA准确率；本次只评估GP阶段吞吐。",
        active_full_run_modified=False, requested_restore_budget=2048)
    write_json(folder / "comparison.json", result)
    for budget, s in arms.items():
        print(f"预算{budget}: 成功{s['success']}/200，{s['wall_seconds']:.1f}秒，{s['jobs_per_second']:.3f}题/秒；"
            f"请求失败{s['request_failed']}，格式失败{s['format_failure']}，格式重试{s['format_retries']}", flush=True)
    print(f"4096/2048吞吐倍数: {speedup:.3f}" if speedup is not None else "吞吐倍数不可用", flush=True)
    print(f"两组图模式完全一致: {same}/200；有效无失败对照: {clean}", flush=True)
    if not clean:
        print("对照包含失败、截断或缺失用量等情况，不能据此认定4096更快。", flush=True)
    print("默认恢复2048；先查看结果，再决定是否调整全量服务参数。", flush=True)


def run(folder, restore_only=False):
    import fcntl
    meta, full, frozen = validate_benchmark(folder)
    with (full / ".full-run.lock").open("a") as full_lock, (folder / "benchmark.lock").open("a") as bench_lock:
        try:
            fcntl.flock(full_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(bench_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("全量程序或测速仍在运行；请先在全量tmux按Ctrl+C，等待其退出。")
        if restore_only:
            restore_service(folder)
            return
        require(not (folder / "started.json").exists(), "该测速已启动过；计时实验不能拼接续跑，请重新prepare创建新目录。")
        service_state(2048)
        jobs = json.loads((folder / "bulk/jobs.json").read_text(encoding="utf-8"))
        warmup = json.loads((folder / "bulk/warmup.json").read_text(encoding="utf-8"))
        write_json(folder / "started.json", dict(started=datetime.now(timezone.utc).isoformat()))
        arms, patterns = {}, {}
        mutated = False
        try:
            for budget in (2048, 4096):
                mutated = True
                print(f"\n=== 新启动{budget}预算服务；加载和预热不计入测速 ===", flush=True)
                guarded_switch(folder, budget)
                state = service_state(budget)
                write_json(folder / f"service-{budget}.json", state)
                snapshot(folder, f"before-{budget}")
                arm_folder = folder / f"arm-{budget}"
                arm_folder.mkdir()
                (arm_folder / ".gitignore").write_text("bulk/\n", encoding="utf-8")
                summary, values = asyncio.run(run_benchmark_arm(jobs, warmup, folder / "gp.txt", budget,
                    arm_folder, frozen.parse_gp_prompt))
                arms[budget], patterns[budget] = summary, values
                snapshot(folder, f"after-{budget}")
            compare(folder, arms, patterns)
        finally:
            if mutated:
                print("\n=== 恢复32B原2048预算服务 ===", flush=True)
                restore_service(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "restore"))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    require(not sys.flags.optimize, "请勿使用-O或PYTHONOPTIMIZE")
    require((PROJECT / "src/utils/query_process.py").is_file(), "请先cd到服务器SSR项目根目录")
    if args.stage == "prepare":
        require(args.run_dir is None, "prepare自动创建独立测速目录")
        prepare()
    else:
        folder = Path(args.run_dir or POINTER.read_text(encoding="utf-8").strip()).resolve()
        def terminate(signum, frame):
            raise KeyboardInterrupt("收到终止信号")
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            run(folder, restore_only=args.stage == "restore")
        finally:
            signal.signal(signal.SIGTERM, previous)




"""Select unique, stratified GP requests without consulting answers or outputs."""

import hashlib
import heapq
import json
import random
from pathlib import Path


def choose_jobs(source, n=200, warmup_count=8, seed=42, expected=54584):
    """Return (jobs, warmups, stats) from a frozen full-run inputs.jsonl.

    Benchmark and warmup requests are disjoint and each is balanced by atype.
    Every unique prompt receives one seeded random priority; the smallest n/2
    priorities in each stratum are benchmark requests and the next warmup/2
    are warmups. Increasing warmup_count leaves benchmark requests unchanged.
    Identical prompts use their first row; identical questions with different
    prompts remain eligible. Returned strings are never normalized or rebuilt.
    Retains at most n + warmup_count prompts, plus one digest per unique input.
    The caller must compare stats['input_sha256'] to the frozen run manifest.
    """
    for name, value, minimum in (("n", n, 2), ("warmup_count", warmup_count, 0),
                                 ("expected", expected, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if n % 2 or warmup_count % 2 or type(seed) is not int:
        raise ValueError("n and warmup_count must be even; seed must be an integer")
    source = Path(source)
    initial = source.stat()
    rng, file_hash, seen = random.Random(seed), hashlib.sha256(), {}
    heaps = {kind: [] for kind in ("entity", "time")}
    row_counts, unique_counts = dict.fromkeys(heaps, 0), dict.fromkeys(heaps, 0)
    per_type, warm_per_type = n // 2, warmup_count // 2
    capacity, count, source_bytes = per_type + warm_per_type, 0, 0
    fields = ("row_index", "question", "qtype", "atype", "prompt", "prompt_sha256")
    with source.open("rb") as stream:
        for index, raw in enumerate(stream):
            count, source_bytes = index + 1, source_bytes + len(raw)
            file_hash.update(raw)
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError) as error:
                raise ValueError(f"row {index}: invalid UTF-8 JSON") from error
            if not isinstance(row, dict) or any(key not in row for key in fields):
                raise ValueError(f"row {index}: missing required input fields")
            if type(row["row_index"]) is not int or row["row_index"] != index:
                raise ValueError(f"row {index}: row_index must be sequential from zero")
            if any(not isinstance(row[key], str) for key in ("question", "atype", "prompt", "prompt_sha256")):
                raise ValueError(f"row {index}: input text fields must be strings")
            kind, prompt, digest = row["atype"], row["prompt"], row["prompt_sha256"]
            if kind not in heaps:
                raise ValueError(f"row {index}: unknown atype {kind!r}")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != digest:
                raise ValueError(f"row {index}: prompt_sha256 mismatch")
            row_counts[kind] += 1
            if digest in seen:
                if seen[digest] != kind:
                    raise ValueError(f"row {index}: duplicate prompt has conflicting atype")
                continue
            seen[digest] = kind
            unique_counts[kind] += 1
            priority = rng.random()
            heap = heaps[kind]
            # Negative priorities put the largest retained priority at heap[0].
            if len(heap) < capacity or (-priority, -index) > heap[0][:2]:
                entry = (-priority, -index, {key: row[key] for key in fields})
                if len(heap) < capacity:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heapreplace(heap, entry)
    final = source.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if stamp(initial) != stamp(final):
        raise ValueError("source inputs changed during selection")
    if count != expected:
        raise ValueError(f"expected {expected} rows, found {count}")
    if any(value < capacity for value in unique_counts.values()):
        raise ValueError(f"need {capacity} unique prompts per atype; found {unique_counts}")
    jobs, warmups = [], []
    for heap in heaps.values():
        ranked = [entry[2] for entry in sorted(heap, reverse=True)]
        jobs.extend(ranked[:per_type])
        warmups.extend(ranked[per_type:])
    jobs.sort(key=lambda job: job["row_index"])
    warmups.sort(key=lambda job: job["row_index"])
    stats = dict(input_sha256=file_hash.hexdigest(), source_bytes=source_bytes,
                 total_rows=count, unique_prompts=len(seen), duplicate_rows=count - len(seen),
                 rows_by_atype=row_counts, unique_by_atype=unique_counts, seed=seed,
                 selected_count=n, warmup_count=warmup_count)
    return jobs, warmups, stats

"""Measured GP inference without application cache; the author's GP parser stays authoritative."""

import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from utils.io_utils import read_prompt
from utils.query_process import get_query_graph_pattern


class TransportFailure(RuntimeError):
    pass


def _usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        key: usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _valid(response):
    matches = [re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL) for tag in ("fact", "time")]
    return all(matches) and all(
        len(match.group(1).strip().split(";")) == size
        for match, size in zip(matches, (3, 2))
    )


def _counts(rows):
    attempts = [attempt for row in rows for attempt in row["attempts"]]
    http_successes = [attempt for attempt in attempts if attempt["http_success"]]
    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    result = {
        key: sum(row[key] for row in rows)
        for key in ("success", "format_failure", "request_failed", "logical_requests", "model_reply_calls")
    }
    result.update(
        total=len(rows), completed=len(rows), job_errors=sum(bool(row["job_error"]) for row in rows),
        transport_tries=len(attempts), http_successful_attempts=len(http_successes),
        transport_failed_attempts=sum(bool(attempt["error"]) for attempt in attempts),
        transport_retries=sum(attempt["transport_attempt"] > 1 for attempt in attempts),
        format_retries=sum(max(0, row["logical_requests"] - 1) for row in rows),
        usage_missing_count=sum(
            attempt["usage"] is None
            or any(not isinstance(attempt["usage"].get(key), int) for key in token_keys)
            for attempt in http_successes
        ),
        truncated=sum(attempt["finish_reason"] == "length" for attempt in http_successes),
        finish_reasons=dict(Counter(str(attempt["finish_reason"] or "unknown") for attempt in http_successes)),
    )
    result["usage_complete"] = result["usage_missing_count"] == 0
    for key in token_keys:
        result[key] = sum(
            attempt["usage"][key]
            for attempt in http_successes
            if attempt["usage"] is not None and isinstance(attempt["usage"].get(key), int)
        )
    latencies = sorted(row["latency_seconds"] for row in rows)
    result.update(
        median_latency_seconds=statistics.median(latencies) if latencies else None,
        p95_latency_seconds=latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
    )
    return result


class _MeasuredClient:
    """One job's metrics, one arm's shared SDK transport, no cache lookup or write."""

    def __init__(self, sdk, alias, prompt):
        self.sdk, self.alias, self.prompt = sdk, alias, prompt
        self.attempts, self.responses, self.logical_requests = [], [], 0

    async def request(self, prompt, model=None, temperature=0.0, max_tokens=2048,
                      stop=None, logprobs=False, use_cache=True):
        if prompt != self.prompt:
            raise ValueError("Actual GP prompt differs from the frozen original")
        if temperature != 0 or max_tokens != 4096 or stop is not None or logprobs:
            raise ValueError("Author GP request sampling parameters changed")
        self.logical_requests += 1
        for retry in range(1, 6):
            attempt = dict(request_index=self.logical_requests, transport_attempt=retry,
                           http_success=False, usage=None, error=None, finish_reason=None)
            self.attempts.append(attempt)
            started = time.perf_counter()
            try:
                response = await self.sdk.chat.completions.create(
                    model=self.alias, messages=[dict(role="user", content=prompt)],
                    temperature=temperature, max_tokens=max_tokens, stop=stop, logprobs=logprobs,
                )
                attempt.update(http_success=True, usage=_usage(response))
                choice = response.choices[0]
                attempt["finish_reason"] = getattr(choice, "finish_reason", None)
                text = choice.message.content
                if not isinstance(text, str):
                    raise ValueError("SDK returned non-text message content")
                self.responses.append(text)
                return text, True
            except Exception as error:
                attempt["error"] = f"{type(error).__name__}: {error}"[:1000]
            finally:
                attempt["elapsed_seconds"] = time.perf_counter() - started
            # Preserve the author's five-attempt, 0.5-second transport retry policy.
            await asyncio.sleep(0.5)
        raise TransportFailure("Five transport attempts failed: " + self.attempts[-1]["error"])


async def run_benchmark_arm(jobs, warmup, template_path, budget, out_dir, parse_gp_prompt,
                            alias="qwen32b-gp", base_url="http://127.0.0.1:8000/v1",
                            client_override=None):
    """Return (summary, prompt-SHA -> (fact tuple, time tuple)); budget labels the server arm.

    Both SDK-owned and supplied clients are closed. Warmups and measured jobs issue real
    requests even for repeated prompts. Job latency starts after acquiring one of 8 slots;
    measured wall time excludes warmup and client closure. The 4096 generation limit is
    independent of the vLLM max-num-batched-tokens budget controlled by the caller.
    """
    sdk = client_override
    try:
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("Server batching budget must be positive")
        jobs, warmup = list(jobs), list(warmup)
        template = read_prompt(str(template_path))
        prepared = []
        for phase in (warmup, jobs):
            entries = []
            for index, job in enumerate(phase):
                prompt = job["prompt"]
                question, facts = parse_gp_prompt(prompt, template)
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if question != job["question"] or template.format(question=question, facts=facts) != prompt:
                    raise ValueError("Job question/facts do not reproduce the frozen GP prompt")
                if job.get("prompt_sha256", digest) != digest:
                    raise ValueError("Frozen GP prompt SHA256 mismatch")
                entries.append((index, job, facts, digest))
            prepared.append(entries)
        destination = Path(out_dir)
        bulk = destination / "bulk"
        bulk.mkdir(parents=True, exist_ok=True)
        if (destination / "summary.json").exists():
            raise FileExistsError("Refusing to replace a completed benchmark arm")
        if sdk is None:
            from openai import AsyncOpenAI
            # Disable hidden SDK retries; count each of the author's transport attempts.
            sdk = AsyncOpenAI(api_key="dummy", base_url=base_url, max_retries=0, timeout=600)
        semaphore = asyncio.Semaphore(8)

        async def run_batch(entries, path, label):
            with path.open("x", encoding="utf-8") as output, tqdm(total=len(entries), desc=label) as bar:
                async def worker(entry):
                    index, job, facts, digest = entry
                    async with semaphore:
                        started = time.perf_counter()
                        client = _MeasuredClient(sdk, alias, job["prompt"])
                        row = dict(job_index=index, row_index=job.get("row_index", index),
                                   question=job["question"], prompt=job["prompt"], prompt_sha256=digest,
                                   fact_pattern=None, time_pattern=None, request_failed=False,
                                   format_failure=False, job_error=None, success=False)
                        try:
                            fp, tp = await get_query_graph_pattern(job["question"], facts, str(template_path), client)
                            row.update(fact_pattern=list(fp), time_pattern=list(tp),
                                       format_failure=not bool(client.responses and _valid(client.responses[-1])))
                            row["success"] = not row["format_failure"]
                        except TransportFailure as error:
                            row.update(request_failed=True, job_error=None, error=str(error))
                        except Exception as error:
                            row["job_error"] = f"{type(error).__name__}: {error}"[:1000]
                        row.update(attempts=client.attempts, responses=client.responses,
                                   response=client.responses[-1] if client.responses else None,
                                   logical_requests=client.logical_requests, model_reply_calls=len(client.responses),
                                   latency_seconds=time.perf_counter() - started)
                        counts = _counts([row])
                        for key in ("transport_tries", "http_successful_attempts", "transport_failed_attempts",
                                    "transport_retries", "format_retries", "usage_missing_count", "truncated",
                                    "finish_reasons"):
                            row[key] = counts[key]
                        row["usage"] = {key: counts[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        bar.update(1)
                        return row

                tasks = [asyncio.create_task(worker(entry)) for entry in entries]
                try:
                    return await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        warm_started = time.perf_counter()
        warm_rows = await run_batch(prepared[0], bulk / "warmup.jsonl", f"GP warmup {budget}")
        warm_seconds = time.perf_counter() - warm_started
        started = time.perf_counter()
        rows = await run_batch(prepared[1], bulk / "results.jsonl", f"GP measured {budget}")
        elapsed = time.perf_counter() - started
        summary = _counts(rows)
        summary.update(
            budget=budget, model=alias, base_url=base_url, concurrency=8, temperature=0, max_tokens=4096,
            sdk_max_retries=0, timeout_seconds=600, application_cache=False,
            wall_seconds=elapsed, unique_prompts=len({row["prompt_sha256"] for row in rows}),
            jobs_per_second=len(rows) / elapsed, successful_jobs_per_second=summary["success"] / elapsed,
            completion_tokens_per_second=summary["completion_tokens"] / elapsed,
            total_tokens_per_second=summary["total_tokens"] / elapsed,
            latency_definition="per job after semaphore acquisition, including format/transport retries",
            p95_method="nearest rank", warmup=dict(_counts(warm_rows), wall_seconds=warm_seconds),
        )
        summary["comparison_eligible"] = bool(rows) and all(
            not data[key] for data in (summary, summary["warmup"])
            for key in ("request_failed", "format_failure", "job_errors", "usage_missing_count",
                        "truncated", "transport_failed_attempts")
        )
        temporary = destination / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination / "summary.json")
        patterns = {
            row["prompt_sha256"]: (tuple(row["fact_pattern"]), tuple(row["time_pattern"]))
            for row in rows if row["success"]
        }
        return summary, patterns
    finally:
        if sdk is not None:
            await sdk.close()


if __name__ == "__main__":
    main()
