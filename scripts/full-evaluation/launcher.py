#!/usr/bin/env python3
"""SSR 全量评测：预检、模型服务、tmux 与独立缓存。"""

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import string
import sys
from pathlib import Path


QWEN_GENERATION_REL = "artifacts/shared/generation-config"


def _pf_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _pf_stat(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'缺少非空文件：{path}')
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def _pf_tensor_header(path):
    size = _pf_stat(path)['size']
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError(f'权重头部损坏：{path}')
        length = struct.unpack('<Q', length_bytes)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, size - 8):
            raise RuntimeError(f'权重头部长度异常：{path}')
        header = json.loads(stream.read(length))
    dtype_bits = dict(BOOL=8, U8=8, I8=8, I16=16, U16=16, I32=32,
                      U32=32, I64=64, U64=64, F16=16, BF16=16,
                      F32=32, F64=64, F8_E4M3=8, F8_E5M2=8)
    spans, keys = [], set()
    if not isinstance(header, dict):
        raise RuntimeError(f'权重头部不是对象：{path}')
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
        bits = dtype_bits.get(tensor.get('dtype'))
        if (not bits or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise RuntimeError(f'权重张量元数据异常：{path}:{name}')
        begin, end = offsets
        if not (0 <= begin <= end <= size - 8 - length
                and end - begin == math.prod(shape) * bits // 8):
            raise RuntimeError(f'权重张量偏移异常：{path}:{name}')
        spans.append((begin, end)); keys.add(name)
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise RuntimeError(f'权重数据存在重叠或空洞：{path}')
        cursor = end
    if not keys or cursor != size - 8 - length:
        raise RuntimeError(f'权重文件长度与张量头不符：{path}')
    return keys


def preflight(root: Path, profile: dict) -> dict:
    import numpy as np
    root = root.resolve()
    small, large = {}, {}

    def checked(relative, hashed=True):
        path = root / relative
        state = _pf_stat(path)
        if hashed:
            small[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            large[str(relative)] = state
        return path

    env_root = Path.home() / 'miniconda3/envs'
    executables = {name: env_root / name / 'bin/python' for name in ('ssr', 'vllm')}
    executables['vllm_cli'] = env_root / 'vllm/bin/vllm'
    for name, path in executables.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f'解释器/工具不可执行：{path}')
    if Path(sys.executable).resolve() != executables['ssr'].resolve():
        raise RuntimeError('请使用 $HOME/miniconda3/envs/ssr/bin/python 启动。')
    for command in ('tmux', 'nvidia-smi', 'ss'):
        if not shutil.which(command):
            raise RuntimeError(f'缺少工具：{command}')
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise RuntimeError('项目所在磁盘可用空间不足3GiB；请先腾出缓存和日志空间。')
    probe = ('import importlib.util as u, importlib.metadata as m,json,sys; '
             'pairs=json.loads(sys.argv[1]); '
             'assert all(u.find_spec(x) for x in pairs), "缺少Python包"; '
             'print(json.dumps({"python":sys.version,"packages":'
             '{x:m.version(y) for x,y in pairs.items()}}))')
    packages = dict(numpy='numpy', torch='torch', transformers='transformers',
                    faiss='faiss-cpu', openai='openai', tqdm='tqdm')
    environments = {}
    for name, pairs in (('ssr', packages), ('vllm', {'vllm': 'vllm', 'torch': 'torch'})):
        if name == 'ssr':
            import importlib.metadata as metadata
            try:
                metadata.version('faiss-cpu')
            except metadata.PackageNotFoundError:
                pairs['faiss'] = 'faiss-gpu'
        result = subprocess.run([str(executables[name]), '-c', probe, json.dumps(pairs)],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'{name} 环境包检查失败：{result.stderr.strip()}')
        environments[name] = json.loads(result.stdout)
    gpu_probe = subprocess.run([str(executables['ssr']), '-c',
        'import torch,faiss,numpy,openai,transformers; '
        'assert torch.cuda.is_available(), "ssr环境无法使用CUDA"; '
        'print(torch.cuda.get_device_name(0))'], text=True, capture_output=True, timeout=60)
    if gpu_probe.returncode:
        raise RuntimeError('ssr核心包/CUDA检查失败：' + gpu_probe.stderr.strip())
    environments['gpu'] = gpu_probe.stdout.strip()
    questions = _pf_json(checked('datasets/MultiTQ/questions/test.json'))
    if not isinstance(questions, list) or len(questions) != 54584:
        raise RuntimeError('全量测试集必须为54584条原始记录。')
    for row in questions:
        question = (row.get('question') or row.get('Question')) if isinstance(row, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError('测试集中存在无题目记录。')
        if not any(key in row for key in ('answers', 'answer', 'Answer')):
            raise RuntimeError('测试集中存在无标准答案的记录。')
    kg = checked('datasets/MultiTQ/kg/full.txt')
    with kg.open(encoding='utf-8') as stream:
        if not any(line.strip() for line in stream):
            raise RuntimeError('知识图谱为空。')
    ids = {}
    for kind in ('entity', 'relation'):
        ids[kind] = _pf_json(checked(f'datasets/MultiTQ/kg/{kind}2id.json'))
        if not isinstance(ids[kind], dict) or not ids[kind]:
            raise RuntimeError(f'{kind}2id必须是非空JSON对象。')
    for kind, fields in (('gp', {'question', 'facts'}), ('qa', {'question', 'facts', 'filter'})):
        path = checked(prompt_path(profile, kind))
        prompt = '\n'.join(line.strip() for line in path.read_text(encoding='utf-8').splitlines())
        templates = prompt.split('\n--- SSR Llama format retry ---\n') if kind == 'qa' else [prompt]
        if len(templates) > 2:
            raise RuntimeError('QA 提示词最多可包含一份格式补答模板。')
        required = ('<fact>', '</fact>', '<time>', '</time>', 'Slot selection:') if kind == 'gp' else ('So the answer is',)
        for template in templates:
            names = {field for _, field, _, _ in string.Formatter().parse(template) if field}
            if names != fields:
                raise RuntimeError(f'{kind}提示词占位符异常：{names}')
            template.format(**{name: 'PRECHECK' for name in fields})
            if any(token not in template for token in required):
                raise RuntimeError(f'{kind}提示词缺少已确认的槽位/输出协议。')
    for filename in ('multitq_text_index.pkl', 'multitq_graph_retriever.pkl'):
        checked('artifacts/shared/indexes/' + filename, False)
    bge = 'models/bge-m3/'
    bge_config = _pf_json(checked(bge + 'config.json'))
    dimension = bge_config.get('hidden_size')
    if type(dimension) is not int or dimension <= 0:
        raise RuntimeError('BGE hidden_size配置异常。')
    for kind in ('entity', 'relation'):
        path = checked(f'artifacts/shared/indexes/MultiTQ_{kind}_embeddings.npy', False)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != (len(ids[kind]), dimension) or array.dtype != np.float32:
            raise RuntimeError(f'{kind}嵌入行数、维度或dtype异常：{array.shape}/{array.dtype}')
        del array
    model = profile['model_rel'].rstrip('/') + '/'
    config = _pf_json(checked(model + 'config.json'))
    quant = config.get('quantization_config') or {}
    if profile.get('model_type') == 'llama':
        if (config.get('model_type') != 'llama'
                or 'LlamaForCausalLM' not in config.get('architectures', [])
                or config.get('hidden_size') != 4096 or config.get('num_hidden_layers') != 32
                or (config.get('rope_scaling') or {}).get('rope_type') != 'llama3'
                or config.get('torch_dtype', config.get('dtype')) != 'bfloat16' or quant):
            raise RuntimeError('所选权重必须为未量化的 Llama 3.1 8B BF16。')
    elif quant.get('quant_method') != 'awq' or quant.get('bits') != 4:
        raise RuntimeError('所选 Qwen 权重必须为 4bit AWQ。')
    for directory in (bge, model):
        tokenizer = _pf_json(checked(directory + 'tokenizer_config.json'))
        if directory == model and not tokenizer.get('chat_template'):
            raise RuntimeError(f'模型缺少 chat_template：{directory}')
        vocabulary = [n for n in ('tokenizer.json', 'sentencepiece.bpe.model', 'tokenizer.model', 'vocab.json', 'merges.txt') if (root / directory / n).is_file()]
        if not vocabulary or ('vocab.json' in vocabulary and 'merges.txt' not in vocabulary):
            raise RuntimeError(f'本地tokenizer文件不完整：{directory}')
        for filename in vocabulary:
            path = checked(directory + filename)
            if filename.endswith('.json'):
                _pf_json(path)
        for filename in ('special_tokens_map.json', 'added_tokens.json'):
            if (root / directory / filename).is_file():
                _pf_json(checked(directory + filename))
    bge_index = next((name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if (root / bge / name).is_file()), None)
    bge_weights = set(_pf_json(checked(bge + bge_index))['weight_map'].values()) if bge_index else {next((name for name in ('model.safetensors', 'pytorch_model.bin') if (root / bge / name).is_file()), 'model.safetensors')}
    if not bge_weights:
        raise RuntimeError('BGE权重索引为空。')
    for name in bge_weights:
        if Path(name).name != name:
            raise RuntimeError('BGE权重索引含非法路径。')
        checked(bge + name, False)
    mapping = _pf_json(checked(model + 'model.safetensors.index.json')).get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f'模型权重索引为空：{model}')
    for shard in sorted(set(mapping.values())):
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise RuntimeError(f'模型权重索引含非法路径：{model}')
        keys = _pf_tensor_header(checked(model + shard, False))
        if keys != {name for name, filename in mapping.items() if filename == shard}:
            raise RuntimeError(f'模型分片张量名与索引不符：{shard}')
    generation_dir = profile.get('generation_rel', QWEN_GENERATION_REL)
    _pf_json(checked(f'{generation_dir}/generation_config.json'))
    for folder, patterns in (('src', ('*.py',)), ('scripts/full-evaluation', ('*.py', '*.sh'))):
        for pattern in patterns:
            for path in sorted((root / folder).rglob(pattern)):
                checked(path.relative_to(root).as_posix())
    print('预检通过：54584题、datasets提示词、共享索引/嵌入、模型结构及环境；未加载GPU模型。', flush=True)
    return dict(profile=profile, sha256=small, assets_stat=large, environments=environments,
                verification='大权重/索引/NPY仅结构与size/mtime校验，非全文件哈希。')


"""Local SSR vLLM service management; caller serializes GPU work with its run lock."""
from pathlib import Path
import json
import os
import re
import signal
import socket
import subprocess
import time
from datetime import datetime
from urllib.request import Request, urlopen

def _proc(pid):
    folder = Path("/proc") / str(pid)
    try:
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder / "stat").read_text().rpartition(")")[2].split()
        argv = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
        return {"pid": pid, "parent": int(fields[1]), "session": int(fields[3]),
                "start": fields[19], "state": fields[0], "argv": argv}
    except (OSError, ValueError, IndexError):
        return None

def _snapshot():
    return {int(p.name): item for p in Path("/proc").iterdir()
            if p.name.isdigit() and (item := _proc(int(p.name))) is not None}

def _alive(saved):
    now = _proc(saved["pid"])
    return now and now["start"] == saved["start"] and now["state"] != "Z"

def _descendants(pid, snapshot):
    found = {pid}
    while True:
        expanded = found | {p for p, item in snapshot.items() if item["parent"] in found}
        if expanded == found:
            return found
        found = expanded

def _api(path="/v1/models", payload=None):
    request = Request("http://127.0.0.1:8000" + path,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.load(response)

def _wanted(root, profile):
    relative = profile["model_rel"]
    if _ALLOWED.get(relative) != profile["alias"]:
        raise RuntimeError("模型目录与本项目登记的服务别名不匹配")
    for key in ("max_seqs", "batched_tokens"):
        if key in profile and (type(profile[key]) is not int or profile[key] < 1):
            raise RuntimeError("无效服务参数: " + key)
    if "max_seqs" not in profile:
        raise RuntimeError("缺少服务参数 max_seqs")
    return (root / relative).resolve()

def _inspect(root, profile):
    wanted = _wanted(root, profile)
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=2):
            pass
    except ConnectionRefusedError:
        pending = []
        for item in _snapshot().values():
            tokens = [a for a in item["argv"] if a and not any(c.isspace() for c in a)]
            if item["state"] != "Z" and any(
                Path(a).name == "vllm" or a.startswith(("vllm.", "VLLM::")) for a in tokens
            ):
                pending.append(item["pid"])
        return {"state": "blocked" if pending else "stopped",
                "reason": "未就绪或无法归属的vLLM进程: " + repr(pending) if pending else "8000端口未启动"}
    try:
        served = _api()["data"]
        roots = {str(Path(item["root"]).resolve()) for item in served}
        allowed = {str((root / rel).resolve()) for rel in _ALLOWED}
        if len(roots) != 1 or not roots <= allowed:
            raise RuntimeError("8000 端口不是本项目登记的单模型服务")
        snapshot = _snapshot()
        owners = set(map(int, re.findall(r"pid=(\d+)", subprocess.check_output(
            ["ss", "-ltnp", "sport = :8000"], text=True, timeout=10))))
        if not owners or not owners <= snapshot.keys():
            raise RuntimeError("无法确认端口进程归当前用户所有")
        families = [(pid, _descendants(pid, snapshot)) for pid in owners]
        families = [(pid, members) for pid, members in families if owners <= members]
        if len(families) != 1:
            raise RuntimeError("端口由多个无法统一归属的进程持有")
        pid, family = families[0]
        target = any(item["id"] == profile["alias"] and
                     Path(item["root"]).resolve() == wanted for item in served)
        return {"state": "target" if target else "other_ssr", "model_root": next(iter(roots)),
                "aliases": [item["id"] for item in served], "pid": pid,
                "start": snapshot[pid]["start"],
                "_members": {p: snapshot[p] for p in family if p in snapshot}}
    except Exception as error:
        return {"state": "blocked", "reason": str(error)}

def observe_service(root, profile):
    """Read-only: does not generate, start, stop, or write anything."""
    root = Path(root).resolve()
    try:
        return {k: v for k, v in _inspect(root, profile).items() if not k.startswith("_")}
    except Exception as error:
        return {"state": "blocked", "reason": str(error)}

def _probe(alias):
    response = _api("/v1/chat/completions", {
        "model": alias, "temperature": 0, "max_tokens": 8,
        "messages": [{"role": "user", "content": "Reply OK."}]})
    choices = response.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型接口没有返回实际生成结果")

def _stop(members, leader, log):
    if leader in members and _alive(members[leader]):
        try:
            os.kill(leader, signal.SIGTERM)
        except ProcessLookupError:
            pass
    start, signalled = time.monotonic(), {leader}
    while True:
        live = [item for item in members.values() if _alive(item)]
        if not live:
            return
        if time.monotonic() - start >= 10:
            for item in live:
                if item["pid"] not in signalled:
                    try:
                        os.kill(item["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    signalled.add(item["pid"])
        if time.monotonic() - start > 90:
            raise RuntimeError("停止服务超时，未强制结束: " + repr([p["pid"] for p in live]))
        time.sleep(1)

def _remember(child, identity, known):
    current = _proc(child.pid)
    if not identity or not current or current["start"] != identity["start"]:
        return
    snapshot = _snapshot()
    members = _descendants(child.pid, snapshot)
    members |= {p for p, item in snapshot.items() if item["session"] == identity["session"]}
    known.update({p: snapshot[p] for p in members if p in snapshot})

def _gpu_free():
    output = subprocess.check_output(["nvidia-smi", "-i", "0",
        "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    return tuple(map(float, output.strip().split(",")))

def ensure_service(root, profile, log):
    """Caller must hold the shared full-evaluation run lock before calling."""
    root = Path(root).resolve()
    wanted = _wanted(root, profile)
    executable = Path.home() / "miniconda3/envs/vllm/bin/vllm"
    generation = root / profile.get("generation_rel", QWEN_GENERATION_REL)
    if not executable.is_file() or not wanted.is_dir() or not (generation / "generation_config.json").is_file():
        raise RuntimeError("vLLM、模型目录或公共生成配置缺失")
    state = _inspect(root, profile)
    if state["state"] == "blocked":
        raise RuntimeError(state["reason"])
    _gpu_free()  # Check the GPU/tool before stopping a healthy existing service.
    if state["state"] in ("target", "other_ssr"):
        log("按当前配置重启本项目模型服务: " + state["model_root"] + " -> " + str(wanted))
        _stop(state["_members"], state["pid"], log)
        after = _inspect(root, profile)
        if after["state"] != "stopped":
            raise RuntimeError("旧服务未完全退出: " + str(observe_service(root, profile)))
    deadline = time.monotonic() + 90
    while True:
        total, free = _gpu_free()
        if free >= total * 0.80:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("显存尚未充分释放，未启动新模型")
        time.sleep(2)
    folder = root / profile["artifact_rel"] / "service"
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / ("vllm-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".log")
    command = [str(executable), "serve", str(wanted), "--served-model-name", profile["alias"],
        "--host", "127.0.0.1", "--port", "8000", "--dtype", profile.get("dtype", "half"),
        "--max-model-len", str(profile.get("max_model_len", 8192)),
        "--max-num-seqs", str(profile["max_seqs"]), "--gpu-memory-utilization", "0.80",
        "--generation-config", str(generation), "--disable-log-requests"]
    if "batched_tokens" in profile:
        command += ["--max-num-batched-tokens", str(profile["batched_tokens"])]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    with logfile.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, env=env, start_new_session=True)
    identity, known = _proc(child.pid), {}
    if identity:
        known[child.pid] = identity
    log("启动 " + profile["alias"] + f"，PID={child.pid}，日志={logfile}")
    try:
        deadline, next_notice = time.monotonic() + 600, 0
        while time.monotonic() < deadline:
            _remember(child, identity, known)
            if child.poll() is not None:
                raise RuntimeError("模型服务启动退出，请查看 " + str(logfile))
            current = _inspect(root, profile)
            if current["state"] == "target" and current["pid"] in known and _alive(known[current["pid"]]):
                try:
                    _probe(profile["alias"])
                except Exception:
                    pass
                else:
                    log("模型实际生成检查通过: " + profile["alias"])
                    return {k: v for k, v in dict(current, reused=False, logfile=str(logfile),
                            command=command, generation_config_verified=True).items()
                            if not k.startswith("_")}
            if time.monotonic() >= next_notice:
                log("等待模型就绪，日志: " + str(logfile))
                next_notice = time.monotonic() + 15
            time.sleep(3)
        raise RuntimeError("模型启动超时，请查看 " + str(logfile))
    except BaseException:
        _remember(child, identity, known)
        try:
            _stop(known, child.pid, log)
            child.wait(timeout=90)
        except Exception as error:
            log("清理本次失败服务时遇到问题: " + str(error))
        raise



import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import traceback
from pathlib import Path

PROFILES = {
    '14': dict(alias='qwen14b', model_rel='models/qwen14b-awq',
               prompt_dir='datasets/MultiTQ/prompt/Qwen',
               artifact_rel='artifacts/qwen2.5-14b-awq', max_seqs=64),
    '32': dict(alias='qwen32b-gp', model_rel='models/qwen32b-awq',
               prompt_dir='datasets/MultiTQ/prompt/Qwen',
               artifact_rel='artifacts/qwen2.5-32b-awq', max_seqs=8,
               batched_tokens=2048),
    'llama31-8b': dict(
        alias='llama31-8b', model_type='llama', dtype='bfloat16',
        model_rel='models/llama-3.1-8b-instruct',
        generation_rel='models/llama-3.1-8b-instruct',
        prompt_dir='datasets/MultiTQ/prompt/Llama',
        artifact_rel='artifacts/llama-3.1-8b-instruct',
        max_seqs=8, max_model_len=16384, batched_tokens=2048,
        session='ssr-llama31-8b-full',
    ),
}
_ALLOWED = {profile["model_rel"]: profile["alias"] for profile in PROFILES.values()}


def say(message):
    print(message, flush=True)


def stamp():
    return dt.datetime.now().isoformat(timespec='seconds')


def write_state(run, state):
    state['updated_at'] = stamp()
    temporary = run / 'run.json.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(run / 'run.json')


def check_processes(root):
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            if (item / 'stat').read_text().rpartition(')')[2].split()[0] == 'Z':
                continue
            arguments = (item / 'cmdline').read_bytes().split(b'\0')
            for raw in arguments:
                argument = os.fsdecode(raw)
                if Path(argument).name != 'run_pipeline.py':
                    continue
                script = Path(argument)
                if not script.is_absolute():
                    script = (item / 'cwd').resolve() / script
                if script.resolve() == root / 'src/run_pipeline.py':
                    raise RuntimeError(f'已有本项目管线在运行，PID={item.name}；本次不启动或切换服务。')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def prompt_path(profile, kind):
    return f'{profile["prompt_dir"]}/{kind}_prompt.txt'


def pipeline_command(root, profile, run):
    return [sys.executable, '-u', str(root / 'src/run_pipeline.py'),
            '--run-text-retrieval', '--run-graph-retrieval', '--run-graph-qa',
            '--kg-path', 'datasets/MultiTQ/kg/full.txt',
            '--qa-path', 'datasets/MultiTQ/questions/test.json',
            '--entity-path', 'datasets/MultiTQ/kg/entity2id.json',
            '--relation-path', 'datasets/MultiTQ/kg/relation2id.json',
            '--dataset-name', 'MultiTQ',
            '--text-index-path', 'artifacts/shared/indexes/multitq_text_index.pkl',
            '--graph-cache-path', 'artifacts/shared/indexes/multitq_graph_retriever.pkl',
            '--text-retriever-model-name', str(root / 'models/bge-m3'),
            '--device', 'cuda:0', '--text-top-k', '100', '--qa-retrieve-cnt', '50',
            '--concurrency', '8', '--random-seed', '42',
            '--graph-pattern-prompt', prompt_path(profile, 'gp'),
            '--qa-prompt', prompt_path(profile, 'qa'),
            '--base-url', 'http://127.0.0.1:8000/v1', '--api-key', 'dummy',
            '--llm-model-name', profile['alias'],
            '--cache-path', str(run / 'llm_cache.jsonl')]


def choose_run(root, profile, resume, fingerprint):
    parent = (root / profile['artifact_rel'] / 'runs').resolve()
    if resume:
        run = Path(resume).expanduser().resolve()
        if run.parent != parent or not run.is_dir():
            raise RuntimeError(f'续跑目录必须位于 {parent} 的下一层。')
        state = json.loads((run / 'run.json').read_text(encoding='utf-8'))
        if state.get('profile') != profile or state.get('fingerprint') != fingerprint:
            raise RuntimeError('模型、源码、数据、提示词或运行资产已变化；请新开运行，不复用旧缓存。')
        if state.get('status') == 'completed':
            raise RuntimeError('该目录已完整完成；如需重跑，请去掉 --resume 新建运行。')
        cache = run / 'llm_cache.jsonl'
        if cache.exists():
            with cache.open(encoding='utf-8') as stream:
                for number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            row = json.loads(line)
                            if (not isinstance(row, dict) or not isinstance(row.get('input'), list)
                                    or 'response' not in row):
                                raise ValueError('record lacks an input list')
                        except ValueError as exc:
                            raise RuntimeError(f'缓存第 {number} 行损坏；保留原文件并先修复。') from exc
        return run, state
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / ('full-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run.mkdir()
    return run, dict(profile=profile, fingerprint=fingerprint, created_at=stamp())


def execute(root, profile, resume):
    lock_path = root / 'artifacts/shared/locks/full-evaluation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有 SSR 全量任务占用运行锁。')
        check_processes(root)
        fingerprint = preflight(root, profile)
        run, state = choose_run(root, profile, resume, fingerprint)
        say(f'结果目录：{run}')
        for key in ('error', 'exit_code', 'metrics', 'finished_at'):
            state.pop(key, None)
        state.update(status='checking_service', command=pipeline_command(root, profile, run))
        write_state(run, state)
        process = None
        try:
            check_processes(root)
            state['service'] = ensure_service(root, profile, say)
            state.update(status='running', started_at=stamp())
            write_state(run, state)
            say('启动全量 54584 题；两阶段均使用 ' + profile['alias'])
            with (run / 'run.log').open('a', encoding='utf-8') as log:
                header = f'\n=== attempt {stamp()} ===\n'
                log.write(header)
                process = subprocess.Popen(state['command'], cwd=root, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                           errors='replace', bufsize=1, pass_fds=(lock.fileno(),))
                loaded = done = False
                metrics = None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                    loaded = loaded or bool(re.search(r'Loaded 54584 questions', line))
                    done = done or '========== Done ==========' in line
                    match = re.search(r'Graph QA Accuracy:\s*([\d.]+)\s*\((\d+)\s*/\s*(\d+)\)', line)
                    if match:
                        metrics = dict(accuracy=float(match[1]), correct=int(match[2]), total=int(match[3]))
                code = process.wait()
            state.update(exit_code=code, metrics=metrics)
            if code or not loaded or not done or not metrics or metrics['total'] != 54584:
                raise RuntimeError(f'管线未通过全量完成核验，退出码={code}；查看 {run / "run.log"}')
            state.update(status='completed', finished_at=stamp())
            write_state(run, state)
            say(f'全量完成：{metrics}；结果目录：{run}')
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         error=str(exc))
            write_state(run, state)
            raise


def main():
    parser = argparse.ArgumentParser(description='SSR 全量检索与问答评测启动器')
    parser.add_argument('model', choices=PROFILES)
    parser.add_argument('--check', action='store_true', help='只读预检，不启停服务、不创建运行目录')
    parser.add_argument('--resume', help='复用本启动器同模型未完成运行的请求缓存')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    profile = PROFILES[args.model]
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                      OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    if args.check:
        check_processes(root)
        preflight(root, profile)
        service = observe_service(root, profile)
        say(service)
        if service['state'] == 'blocked':
            raise RuntimeError(service['reason'])
        say('静态预检通过；正式启动时另做服务和实际生成检查。尚未启动全量任务。')
    elif args.worker:
        execute(root, profile, args.resume)
    else:
        session = profile.get('session', f'ssr-{args.model}b-full')
        if subprocess.run(['tmux', 'has-session', '-t', '=' + session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            panes = subprocess.check_output(['tmux', 'list-panes', '-s', '-t', '=' + session,
                                              '-F', '#{pane_dead}'], text=True).splitlines()
            if panes and all(value == '1' for value in panes):
                subprocess.run(['tmux', 'kill-session', '-t', '=' + session], check=True)
            else:
                raise RuntimeError(f'tmux 会话 {session} 仍在运行，请进入查看：tmux attach -t {session}')
        worker = [sys.executable, '-u', str(Path(__file__).resolve()), args.model, '--worker']
        if args.resume:
            worker += ['--resume', str(Path(args.resume).expanduser().resolve())]
        result = subprocess.check_output(['tmux', 'new-session', '-d', '-s', session, '-c', str(root),
                                          '-P', '-F', '#{pane_id} #{window_id}', 'sleep 86400'], text=True)
        pane, window = result.strip().split()
        try:
            subprocess.run(['tmux', 'set-window-option', '-t', window, 'remain-on-exit', 'on'], check=True)
            subprocess.run(['tmux', 'respawn-pane', '-k', '-t', pane, shlex.join(worker)], check=True)
        except BaseException:
            subprocess.run(['tmux', 'kill-session', '-t', '=' + session])
            raise
        say(f'已创建启动会话，预检结果在会话内：tmux attach -t {session}')
        say('分离会话：Ctrl+b，然后按 d。退出状态保留在窗格及结果目录 run.json。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        say('已中断。')
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
