import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

if sys.flags.optimize:
    raise SystemExit("请使用不带-O、未设置PYTHONOPTIMIZE的Python运行")
PROJECT = Path.cwd().resolve()
PILOT = PROJECT / "artifacts/model-gp-ab-fcr5qfjy"
EXPECTED = 54584
sys.path.insert(0, str(PROJECT / "src"))
from tqdm import tqdm
from utils.io_utils import load_qas, read_prompt
from utils.query_process import get_query_graph_pattern, get_qa_result
from utils.eval_utils import evaluate_qa, print_qa_metrics, is_correct
from utils.sample_utils import uniform_sample_preserve_edges

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def require(value, message):
    if not value:
        raise RuntimeError(message)

def progress(run, phase, completed, total, **extra):
    atomic_json(run / f"progress-{phase}.json", dict(phase=phase, completed=completed,
        total=total, updated=datetime.now(timezone.utc).isoformat(), **extra))

def assets():
    files = [
        "src/config.py", "src/run_pipeline.py", "src/utils/client.py", "src/utils/io_utils.py",
        "src/utils/query_process.py", "src/utils/eval_utils.py", "src/utils/sample_utils.py",
        "src/retriever/graph_retriever.py", "src/retriever/text_retriever.py",
        "datasets/MultiTQ/questions/test.json", "datasets/MultiTQ/kg/full.txt",
        "datasets/MultiTQ/kg/entity2id.json", "datasets/MultiTQ/kg/relation2id.json",
        "artifacts/multitq_graph_retriever.pkl", "MultiTQ_entity_embeddings.npy",
        "MultiTQ_relation_embeddings.npy", "models/bge-m3/config.json",
    ]
    for model in ("qwen32b-awq",):
        files.extend(f"models/{model}/{name}" for name in (
            "config.json", "model.safetensors.index.json", "tokenizer_config.json", "generation_config.json"))
    files.append("artifacts/model-gp-ab-fcr5qfjy/sampling-defaults/generation_config.json")
    return {str(PROJECT / name): sha(PROJECT / name) for name in files}

def weight_stats():
    result = {}
    for folder in ("bge-m3", "qwen32b-awq"):
        for pattern in ("*.safetensors", "*.bin"):
            for path in sorted((PROJECT / "models" / folder).glob(pattern)):
                stat = path.stat()
                result[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return result

def verify(run):
    meta = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    require(meta["n"] == EXPECTED, "记录总数不是54584")
    require(json_sha({k: v for k, v in meta.items() if k != "signature"}) == meta["signature"], "清单校验失败")
    require(sha(Path(__file__)) == meta["runner_sha256"], "运行脚本已变化，停止混合续跑")
    for name, digest in meta["assets"].items():
        require(sha(name) == digest, f"输入/代码/索引文件已变化: {name}")
    require(weight_stats() == meta["weight_stats"], "模型权重文件状态已变化")
    require({p: importlib.metadata.version(p) for p in meta["environment"]} == meta["environment"], "SSR运行环境版本已变化")
    require(sha(run / "gp.txt") == meta["gp_template_sha256"] and sha(run / "qa.txt") == meta["qa_template_sha256"], "冻结提示词已变化")
    require(sha(run / "bulk/inputs.jsonl") == meta["input_sha256"], "冻结输入已变化")
    return meta

def prepare(run):
    if (run / "manifest.json").exists():
        meta = verify(run)
        print("已准备完整输入:", meta["n"], flush=True)
        return meta
    (run / ".gitignore").write_text("bulk/\n*.tmp\n.full-run.lock\n", encoding="utf-8")
    (run / "bulk").mkdir(exist_ok=True)
    questions, answers, qtypes, atypes = load_qas(str(PROJECT / "datasets/MultiTQ/questions/test.json"))
    require(len(questions) == EXPECTED, f"测试集记录数不符: {len(questions)}")
    template = read_prompt(str(PILOT / "gp-B.txt"))
    (run / "gp.txt").write_text(template, encoding="utf-8")
    (run / "qa.txt").write_text(read_prompt(str(PROJECT / "artifacts/gp-B-full-MqkyPV/qa.txt")), encoding="utf-8")
    print("校验并恢复54584条原始输入；保持重复题和原顺序……", flush=True)
    frozen = freeze_inputs(run, questions, answers, qtypes, atypes, template,
        PROJECT / "artifacts/gp-B-full-MqkyPV/llm_cache.jsonl")
    frozen["unique_questions"] = frozen.pop("unique")
    frozen["duplicates"] = frozen.pop("duplicatescount")
    meta = dict(version="SSR-GP32-QA32-full-v1", **frozen,
        created=datetime.now(timezone.utc).isoformat(), assets=assets(), weight_stats=weight_stats(),
        runner_sha256=sha(Path(__file__)), gp_template_sha256=sha(run / "gp.txt"),
        qa_template_sha256=sha(run / "qa.txt"), models=dict(gp="qwen32b-gp", qa="qwen32b-gp"),
        temperature=0, max_tokens=4096, concurrency=8,
        environment={p: importlib.metadata.version(p) for p in ("openai", "torch", "transformers", "numpy", "tqdm")})
    meta["signature"] = json_sha(meta)
    atomic_json(run / "manifest.json", meta)
    print(f"输入准备完成: {meta['n']}条；重复记录保留；目录: {run}", flush=True)
    return meta

def records_path(run, phase):
    return run / "bulk" / {"gp": "gp.jsonl", "evidence": "qa-inputs.jsonl", "qa": "qa.jsonl"}[phase]

def finished(run, phase, meta):
    marker = run / f"{phase}-done.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    require(saved["signature"] == meta["signature"] and saved["n"] == meta["n"], "完成标记不匹配")
    require(sha(records_path(run, phase)) == saved["sha256"], f"已完成的{phase}结果已变化")
    return True

def checkpoint(run, phase, meta):
    return load_checkpoint(records_path(run, phase), meta["signature"], phase, meta["n"])

def mark_done(run, phase, meta, rows):
    require(set(rows) == set(range(meta["n"])) and all(not r.get("request_failed", False) for r in rows.values()), f"{phase}未完整覆盖原始记录")
    path = records_path(run, phase)
    with path.open("ab") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(run / f"{phase}-done.json", dict(signature=meta["signature"], n=meta["n"], sha256=sha(path)))
    progress(run, phase, meta["n"], meta["n"], state="complete")

class TransportFailure(RuntimeError):
    pass

class CheckedClient:
    def __init__(self, client, prompt):
        self.client, self.prompt, self.responses = client, prompt, []
    async def request(self, **kwargs):
        require(kwargs["prompt"] == self.prompt, "实际请求与冻结提示词不同")
        response, ok = await self.client.request(**kwargs)
        if not ok:
            raise TransportFailure("服务请求经重试仍失败")
        self.responses.append(response)
        return response, ok

async def model_phase(run, phase, client_override=None, runtime_check=True):
    meta = verify(run)
    if finished(run, phase, meta):
        return
    previous = checkpoint(run, phase, meta)
    done = {i: r for i, r in previous.items() if not r.get("request_failed", False)}
    if len(done) == meta["n"]:
        mark_done(run, phase, meta, done)
        return
    model = "qwen32b-gp"
    model_root = PROJECT / "models/qwen32b-awq"
    cache = run / "bulk" / f"{phase}-cache.jsonl"
    if cache.exists():
        for _ in iter_jsonl(cache, repair_tail=True):
            pass
    if client_override is None:
        from utils.client import AsyncLLMClient
        client = AsyncLLMClient(api_key="dummy", base_url="http://127.0.0.1:8000/v1", model_name=model, cache_path=str(cache))
    else:
        client = client_override
    source = run / "bulk/inputs.jsonl" if phase == "gp" else records_path(run, "evidence")
    if phase == "qa":
        require(finished(run, "evidence", meta), "QA证据尚未完整冻结")
    queue = asyncio.Queue(maxsize=16)
    bar = tqdm(total=meta["n"], initial=len(done), desc=f"Full {phase}")
    async def producer():
        for job in iter_jsonl(source):
            if job["row_index"] not in done:
                await queue.put(job)
        for _ in range(8):
            await queue.put(None)
    async def worker():
        while True:
            job = await queue.get()
            if job is None:
                return
            idx = job["row_index"]
            checked = CheckedClient(client, job["prompt"])
            record = dict(row_index=idx, question=job["question"], phase=phase,
                signature=meta["signature"], prompt_sha256=job["prompt_sha256"], model=model)
            try:
                if phase == "gp":
                    question, facts = parse_gp_prompt(job["prompt"], read_prompt(str(run / "gp.txt")))
                    require(question == job["question"], "问题匹配失败")
                    fp, tp = await get_query_graph_pattern(question, facts, str(run / "gp.txt"), checked)
                    response = checked.responses[-1]
                    fm, tm = re.search(r"<fact>(.*?)</fact>", response, re.S), re.search(r"<time>(.*?)</time>", response, re.S)
                    valid = bool(fm and tm and len(fm[1].strip().split(";")) == 3 and len(tm[1].strip().split(";")) == 2)
                    record.update(fact_pattern=fp, time_pattern=tp, response=response, format_failure=not valid)
                else:
                    prediction, _ = await get_qa_result(job["prompt"], checked)
                    response = checked.responses[-1]
                    record.update(prediction=prediction, response=response, missing_marker="So the answer is" not in response)
                record.update(request_failed=False, request_attempts=len(checked.responses))
            except TransportFailure as error:
                record.update(request_failed=True, error=str(error))
                append_record(records_path(run, phase), record)
                progress(run, phase, len(done), meta["n"], state="request_failed", row_index=idx)
                raise
            append_record(records_path(run, phase), record)
            done[idx] = record
            bar.update(1)
            if len(done) % 100 == 0:
                progress(run, phase, len(done), meta["n"], state="running")
    tasks = []
    try:
        if runtime_check:
            models = await client.async_client.models.list()
            served = [m for m in models.data if m.id == model]
            actual_root = getattr(served[0], "root", None) if len(served) == 1 else None
            require(actual_root and Path(actual_root).resolve() == model_root.resolve(), f"接口实际模型不是{model}")
            atomic_json(run / f"runtime-{phase}.json", dict(model=model, api_root=actual_root,
                hostname=socket.gethostname(), temperature=0, max_tokens=4096))
        tasks = [asyncio.create_task(worker()) for _ in range(8)] + [asyncio.create_task(producer())]
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await client.async_client.close()
        bar.close()
        if len(done) < meta["n"]:
            progress(run, phase, len(done), meta["n"], state="paused")
    mark_done(run, phase, meta, done)
    print(f"{phase}阶段完成: {len(done)}/{meta['n']}", flush=True)

def freeze_evidence(run, graph_override=None):
    meta = verify(run)
    if finished(run, "evidence", meta):
        return
    require(finished(run, "gp", meta), "GP阶段尚未完成")
    gp = checkpoint(run, "gp", meta)
    done = checkpoint(run, "evidence", meta)
    own_graph = graph_override is None
    if own_graph:
        from run_pipeline import build_graph_retriever
        graph = build_graph_retriever(SimpleNamespace(graph_cache_path=str(PROJECT / "artifacts/multitq_graph_retriever.pkl"),
            force_reload=False, text_retriever_model_name=str(PROJECT / "models/bge-m3"), device="cuda:0"))
    else:
        graph = graph_override
    gp_template, qa_template = read_prompt(str(run / "gp.txt")), read_prompt(str(run / "qa.txt"))
    bar = tqdm(total=meta["n"], initial=len(done), desc="Freeze QA evidence")
    try:
        for job in iter_jsonl(run / "bulk/inputs.jsonl"):
            idx = job["row_index"]
            if idx in done:
                continue
            pattern = gp[idx]
            fp, tp = tuple(pattern["fact_pattern"]), tuple(pattern["time_pattern"])
            found = graph.get_facts(fp, tp)
            texts = graph.facts_to_text(found)
            if not texts:
                _, facts = parse_gp_prompt(job["prompt"], gp_template)
                facts = facts[:50]
            elif len(texts) >= 50:
                facts = uniform_sample_preserve_edges(texts, sample_size=50)
            else:
                facts = texts
            condition = " ".join(fp) + f", from {tp[0]} to {tp[1]}"
            prompt = qa_template.format(question=job["question"], facts=facts, filter=condition)
            record = dict(row_index=idx, question=job["question"], phase="evidence", signature=meta["signature"],
                prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), graph_count=len(found),
                qa_count=len(facts), text_fallback=not texts, fact_pattern=fp, time_pattern=tp)
            append_record(records_path(run, "evidence"), record)
            done[idx] = record
            bar.update(1)
            if len(done) % 100 == 0:
                progress(run, "evidence", len(done), meta["n"], state="running")
    finally:
        bar.close()
        del graph
        gc.collect()
        if own_graph:
            import torch
            torch.cuda.empty_cache()
    mark_done(run, "evidence", meta, done)

def score(run):
    meta = verify(run)
    require(finished(run, "qa", meta), "QA尚未完整完成，不发布全量准确率")
    rows = checkpoint(run, "qa", meta)
    require(set(rows) == set(range(meta["n"])) and not any(r["request_failed"] for r in rows.values()), "QA结果不完整")
    qs, answers, qtypes, atypes = load_qas(str(PROJECT / "datasets/MultiTQ/questions/test.json"))
    require(all(rows[i]["question"] == q for i, q in enumerate(qs)), "结果行顺序或问题不匹配")
    predictions = [rows[i]["prediction"] for i in range(len(qs))]
    metrics = evaluate_qa(answers, predictions, qtypes, atypes)
    print_qa_metrics("Graph QA", metrics)
    atomic_json(run / "summary.json", dict(configuration="32B GP + 32B QA", total_records=len(qs),
        metrics=asdict(metrics), missing_marker=sum(r["missing_marker"] for r in rows.values()), signature=meta["signature"]))
    examples, per_type = [], Counter()
    for i, (q, answer, kind, prediction) in enumerate(zip(qs, answers, atypes, predictions)):
        if not is_correct(answer, prediction) and per_type[kind] < 10:
            examples.append(dict(row_index=i, question=q, answer=answer, answer_type=kind,
                prediction=prediction, response_tail=rows[i]["response"][-1200:]))
            per_type[kind] += 1
    atomic_json(run / "error-examples.json", examples)
    print("完整评测完成；全部原始记录已计分。结果目录:", run, flush=True)

def execute(run):
    meta = prepare(run)
    need_gp = not finished(run, "gp", meta)
    need_qa = not finished(run, "qa", meta)
    if need_gp or need_qa:
        subprocess.run([sys.executable, str(PILOT / "switch_model.py"), "32"], check=True)
    if need_gp:
        asyncio.run(model_phase(run, "gp"))
    freeze_evidence(run)
    if need_qa:
        asyncio.run(model_phase(run, "qa"))
    score(run)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("all", "prepare", "gp", "evidence", "qa", "status"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    require(run.parent == (PROJECT / "artifacts").resolve() and run.name.startswith("qwen32b-full-"), "请使用本次32B独立全量目录")
    run.mkdir(exist_ok=True)
    if args.stage == "status":
        for name in ("manifest.json", "progress-gp.json", "progress-evidence.json", "progress-qa.json", "summary.json"):
            if (run / name).exists():
                value = json.loads((run / name).read_text(encoding="utf-8"))
                if name == "manifest.json":
                    value = {k: value[k] for k in ("n", "unique_questions", "duplicates", "duplicate_question_rows", "signature")}
                print(name, json.dumps(value, ensure_ascii=False))
        return
    import fcntl
    with (run / ".full-run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "all": execute(run)
        elif args.stage == "prepare": prepare(run)
        elif args.stage in ("gp", "qa"):
            asyncio.run(model_phase(run, args.stage))
            if args.stage == "qa": score(run)
        elif args.stage == "evidence": freeze_evidence(run)

import json
import os
from pathlib import Path


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def iter_jsonl(path, repair_tail=False):
    """Read strictly; tail repair requires the caller's exclusive writer lock."""
    path = Path(path)
    with path.open("r+b" if repair_tail else "rb") as stream:
        number = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return
            number += 1
            terminated = line.endswith(b"\n")
            try:
                row = json.loads(line.decode("utf-8"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, ValueError) as error:
                if repair_tail and not terminated:
                    stream.truncate(start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    return
                raise ValueError(f"Invalid JSONL: {path}, line {number}, byte {start}") from error
            if repair_tail and not terminated:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            yield row


def load_checkpoint(path, signature, phase, n):
    path = Path(path)
    if not path.exists():
        return {}
    latest = {}
    for row in iter_jsonl(path, repair_tail=True):
        if not isinstance(row, dict) or "signature" not in row or "phase" not in row:
            raise ValueError(f"Invalid checkpoint record: {path}")
        if row["signature"] != signature or row["phase"] != phase:
            raise ValueError(f"Checkpoint signature/phase mismatch: {path}")
        index = row.get("row_index")
        if type(index) is not int or not 0 <= index < n:
            raise ValueError(f"Checkpoint row_index out of range: {index!r}")
        latest[index] = row
    return latest


def append_record(path, row):
    """Synchronous append/close; caller serializes writers and decides completion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")

import ast
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def parse_gp_prompt(prompt, template):
    if template.count("{question}") != 1 or template.count("{facts}") != 1:
        raise ValueError("GP模板必须各包含一个question和facts字段")
    prefix, rest = template.split("{question}", 1)
    middle, suffix = rest.split("{facts}", 1)
    if not middle or not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise ValueError("GP请求与冻结模板不符")
    body = prompt[len(prefix):len(prompt)-len(suffix)] if suffix else prompt[len(prefix):]
    question, separator, raw_facts = body.rpartition(middle)
    if not separator:
        raise ValueError("GP请求缺少问题与事实分隔符")
    facts = ast.literal_eval(raw_facts)
    if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
        raise ValueError("GP事实必须为字符串列表")
    return question, facts


def freeze_inputs(run, questions, answers, qtypes, atypes, template, sourcecache):
    run, source = Path(run), Path(sourcecache)
    bulk, issues_path = run / "bulk", run / "input-issues.json"
    if not (len(questions) == len(answers) == len(qtypes) == len(atypes)):
        raise ValueError("测试集各字段记录数不同")
    if not bulk.is_dir():
        raise ValueError("调用方必须先创建bulk目录及其Git忽略规则")
    target, temporary = bulk / "inputs.jsonl", bulk / "inputs.jsonl.tmp"
    prefix, rest = template.split("{question}", 1)
    _, suffix = rest.split("{facts}", 1)
    wanted, entries, conflicts = set(questions), {}, set()
    issues = dict(missing_count=0, conflict_count=0, malformed_cache_count=0,
                  malformed_gp_count=0, source_changed=False, samples={})
    def sample(kind, value):
        values = issues["samples"].setdefault(kind, [])
        if len(values) < 20:
            values.append(value)
    def report():
        issues_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    def stamp():
        s = source.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns
    initial, source_hash, offset = stamp(), hashlib.sha256(), 0
    with source.open("rb") as stream:
        for raw in stream:
            source_hash.update(raw)
            at, offset = offset, offset + len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                key = row["input"]
                if not isinstance(key, list) or len(key) < 2:
                    raise ValueError("缓存键格式不符")
                prompt, model = key[:2]
                if not isinstance(prompt, str):
                    raise ValueError("缓存提示词不是字符串")
            except (ValueError, KeyError, TypeError) as error:
                issues["malformed_cache_count"] += 1
                sample("cache", dict(offset=at, error=str(error)[:300]))
                continue
            if model != "qwen14b" or not prompt.startswith(prefix) or not prompt.endswith(suffix):
                continue
            try:
                question, _ = parse_gp_prompt(prompt, template)
            except (ValueError, SyntaxError, TypeError) as error:
                issues["malformed_gp_count"] += 1
                sample("gp", dict(offset=at, error=str(error)[:300]))
                continue
            if question not in wanted:
                continue
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if question in entries and entries[question][2] != digest:
                conflicts.add(question)
                sample("conflicts", dict(question=question, first_sha256=entries[question][2], other_sha256=digest))
            else:
                entries.setdefault(question, (at, len(raw), digest))
    missing = sorted(wanted - entries.keys())
    issues.update(missing_count=len(missing), conflict_count=len(conflicts), source_changed=stamp() != initial)
    issues["samples"]["missing"] = missing[:20]
    report()
    if any(issues[k] for k in ("missing_count", "conflict_count", "malformed_cache_count", "malformed_gp_count", "source_changed")):
        raise RuntimeError(f"完整GP输入恢复未通过，请检查 {issues_path}")
    input_hash = hashlib.sha256()
    try:
        with source.open("rb") as stream, temporary.open("wb") as output:
            for index, (q, answer, qtype, atype) in enumerate(zip(questions, answers, qtypes, atypes)):
                at, length, digest = entries[q]
                stream.seek(at)
                prompt = json.loads(stream.read(length))["input"][0]
                if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != digest:
                    raise ValueError(f"第二遍扫描提示词校验失败: row_index={index}")
                row = dict(row_index=index, question=q, answer=answer, qtype=qtype,
                           atype=atype, prompt=prompt, prompt_sha256=digest)
                raw = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                output.write(raw)
                input_hash.update(raw)
            output.flush()
            os.fsync(output.fileno())
        if stamp() != initial:
            issues["source_changed"] = True
            raise ValueError("恢复过程中源缓存发生变化")
    except Exception as error:
        issues["write_error"] = str(error)[:500]
        report()
        raise
    if target.exists():
        existing = hashlib.sha256()
        with target.open("rb") as previous:
            for block in iter(lambda: previous.read(1024 * 1024), b""):
                existing.update(block)
        if existing.hexdigest() != input_hash.hexdigest():
            raise ValueError("未完成准备的输入与重新恢复结果不一致，停止覆盖")
        temporary.unlink()
    else:
        temporary.replace(target)
    frequency = Counter(questions)
    return dict(n=len(questions), unique=len(wanted), input_path=str(target),
                source_sha256=source_hash.hexdigest(), input_sha256=input_hash.hexdigest(),
                template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
                duplicatescount=len(questions)-len(wanted),
                duplicate_question_rows=sum(n for n in frequency.values() if n > 1),
                source_size=initial[2], source_mtime_ns=initial[3])


if __name__ == "__main__":
    main()
