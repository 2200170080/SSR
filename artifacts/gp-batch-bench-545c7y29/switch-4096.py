import hashlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("请使用未设置PYTHONOPTIMIZE且不带-O的Python执行")

arm = sys.argv[1]
assert arm in {"14", "32"}
project = Path.cwd().resolve()
run = project / "artifacts/model-gp-ab-fcr5qfjy"
common = run / "sampling-defaults"
launcher = Path.home() / "miniconda3/envs/vllm/bin/vllm"
spec = {
    "14": dict(alias="qwen14b", path=project / "models/qwen14b-awq", seqs=64),
    "32": dict(alias="qwen32b-gp", path=project / "models/qwen32b-awq", seqs=8),
}

for value in spec.values():
    value["path"] = value["path"].resolve()
    for name in (
        "config.json", "tokenizer_config.json",
        "model.safetensors.index.json"
    ):
        assert (value["path"] / name).is_file(), f"模型文件缺失: {value['path'] / name}"

assert launcher.is_file() and os.access(launcher, os.X_OK)
defaults = json.loads(
    (common / "generation_config.json").read_text(encoding="utf-8")
)
digest = hashlib.sha256(json.dumps(
    defaults, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")
).encode()).hexdigest()

baseline = json.loads(
    (run / "GP14/metadata.json").read_text(encoding="utf-8")
)
assert digest == baseline["common_generation_sha256"], "公共生成配置与14B基线不符"
assert hashlib.sha256(
    (run / "run_gp.py").read_bytes()
).hexdigest() == baseline["runner_sha256"], "GP脚本已变化"
assert hashlib.sha256(
    (project / "src/utils/client.py").read_bytes()
).hexdigest() == baseline["client_sha256"], "客户端已变化"

summary = json.loads(
    (run / "GP14/summary.json").read_text(encoding="utf-8")
)
assert sum(x["total"] for x in summary.values()) == 200
assert all(
    x["request_failed"] == x["format_failure"] == 0
    for x in summary.values()
)

weights = json.loads(
    (run / "weights32-check.json").read_text(encoding="utf-8")
)
for name, key in (
    ("config.json", "config_sha256"),
    ("model.safetensors.index.json", "index_sha256")
):
    assert hashlib.sha256(
        (spec["32"]["path"] / name).read_bytes()
    ).hexdigest() == weights[key]

switch_lock = (run / ".service-switch.lock").open("a")
try:
    fcntl.flock(switch_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("已有模型切换任务执行中，不重复启动")

def port_open():
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            return True
    except OSError:
        return False

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)

def process(pid):
    try:
        folder = Path("/proc") / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder / "stat").read_text().rpartition(")")[2].split()
        argv = (
            (folder / "cmdline").read_bytes()
            .decode(errors="replace").strip("\0").split("\0")
        )
        return dict(
            pid=pid, parent=int(fields[1]), start=fields[19],
            state=fields[0], argv=argv
        )
    except (OSError, ValueError, IndexError):
        return None

def processes():
    return {
        int(p.name): info
        for p in Path("/proc").iterdir()
        if p.name.isdigit()
        and (info := process(int(p.name))) is not None
    }

def gpu_pids():
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"
    ], capture_output=True, text=True, check=True)
    return {
        int(line.strip()) for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }

def service_candidates(snapshot, definitions):
    found = []
    for info in snapshot.values():
        argv = info["argv"]
        if "serve" not in argv:
            continue
        at = argv.index("serve")
        if (
            not any(Path(a).name == "vllm" for a in argv[:at])
            or at + 1 >= len(argv)
        ):
            continue
        model_path = Path(argv[at + 1]).expanduser().resolve()
        port = (
            int(argv[argv.index("--port") + 1])
            if "--port" in argv else 8000
        )
        port = next((
            int(a.split("=", 1)[1]) for a in argv
            if a.startswith("--port=")
        ), port)
        for key, value in definitions.items():
            if model_path == value["path"] and port == 8000:
                found.append((key, info))
    return found

try:
    served = request("/v1/models")["data"]
except Exception:
    served = None

all_processes = processes()
candidates = service_candidates(all_processes, spec)

for key in spec:
    tracked = run / f"service-{key}.json"
    if not tracked.is_file():
        continue
    saved = json.loads(tracked.read_text(encoding="utf-8"))
    info = all_processes.get(saved["pid"])
    if info and info["state"] != "Z" and info["start"] == saved["start"]:
        assert (
            saved["model"] == spec[key]["alias"]
            and Path(saved["root"]).resolve() == spec[key]["path"]
        )
        assert saved["generation_sha256"] == digest, "运行服务的生成配置记录不符"
        if not any(item[1]["pid"] == info["pid"] for item in candidates):
            candidates.append((key, info))

assert len(candidates) <= 1, "无法唯一识别本项目vLLM主进程，未停止任何进程"
target = spec[arm]

if False and served and any(
    m["id"] == target["alias"]
    and Path(m["root"]).resolve() == target["path"]
    for m in served
):
    if arm == "32":
        saved = json.loads(
            (run / "service-32.json").read_text(encoding="utf-8")
        )
        assert len(candidates) == 1 and candidates[0][0] == arm
        assert (
            candidates[0][1]["pid"] == saved["pid"]
            and candidates[0][1]["start"] == saved["start"]
        )
        assert saved["generation_sha256"] == digest
    result = request("/v1/chat/completions", dict(
        model=target["alias"],
        messages=[dict(role="user", content="Reply OK.")],
        temperature=0, max_tokens=1
    ))
    assert result.get("choices")
    print("目标模型已经就绪:", target["alias"])
    sys.exit(0)

if candidates:
    old, info = candidates[0]
    if served:
        assert any(
            m["id"] == spec[old]["alias"]
            and Path(m["root"]).resolve() == spec[old]["path"]
            for m in served
        ), "接口与进程不一致"
    else:
        tracked = run / f"service-{old}.json"
        assert tracked.is_file(), "服务未就绪且无已知启动记录，未停止任何进程"
        recorded = json.loads(tracked.read_text(encoding="utf-8"))
        assert (
            recorded["pid"] == info["pid"]
            and recorded["start"] == info["start"]
        ), "启动记录与进程不符"

    all_processes = processes()
    refreshed = all_processes.get(info["pid"])
    assert refreshed and refreshed["start"] == info["start"], "进程身份已变化"
    info = refreshed
    tree = {info["pid"]}
    while True:
        expanded = tree | {
            pid for pid, item in all_processes.items()
            if item["parent"] in tree
        }
        if expanded == tree:
            break
        tree = expanded

    old_gpu = gpu_pids() & tree
    current = process(info["pid"])
    assert current and current["start"] == info["start"], "进程身份已变化"

    print(
        f"停止本项目 {spec[old]['alias']}，PID={info['pid']}",
        flush=True
    )
    os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 120
    while True:
        alive = [
            pid for pid in tree
            if (now := process(pid)) and now["state"] != "Z"
            and now["start"] == all_processes[pid]["start"]
        ]
        if not port_open() and not alive and not (gpu_pids() & old_gpu):
            break
        assert time.monotonic() < deadline, (
            "旧服务或工作进程尚未释放；已停止切换，不会强制终止"
        )
        time.sleep(2)
else:
    assert not port_open(), "8000端口由未知服务占用，未作修改"
    assert not any(
        p["state"] != "Z"
        and any("vllm" in arg.lower() for arg in p["argv"][:2])
        for p in all_processes.values()
    ), "仍有无法归属的vLLM进程，未启动新服务"

memory = subprocess.check_output([
    "nvidia-smi", "-i", "0",
    "--query-gpu=memory.total,memory.free",
    "--format=csv,noheader,nounits"
], text=True)
total, free = [float(x.strip()) for x in memory.strip().split(",")]
assert free >= total * 0.80, (
    f"显存未充分释放: free={free} MiB；未启动新模型"
)

log = run / f"vllm-{arm}-{datetime.now():%Y%m%d-%H%M%S}.log"
command = [
    str(launcher), "serve", str(target["path"]),
    "--served-model-name", target["alias"],
    "--host", "127.0.0.1", "--port", "8000",
    "--dtype", "half", "--max-model-len", "8192",
    "--max-num-seqs", str(target["seqs"]),
    "--max-num-batched-tokens", "4096",
    "--gpu-memory-utilization", "0.80",
    "--generation-config", str(common),
    "--disable-log-requests",
]
env = os.environ.copy()
env.update(
    CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8",
    MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"
)
with log.open("w", encoding="utf-8") as output:
    child = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=output,
        stderr=subprocess.STDOUT, start_new_session=True, env=env
    )

identity = process(child.pid)
(run / f"service-{arm}.json").write_text(json.dumps(dict(
    pid=child.pid,
    start=identity["start"] if identity else None,
    model=target["alias"], root=str(target["path"]),
    command=command, log=str(log), generation_sha256=digest
), indent=2), encoding="utf-8")

print(
    "启动:", target["alias"], "PID:", child.pid,
    "日志:", log, flush=True
)
deadline = time.monotonic() + 600
last_status = 0

while time.monotonic() < deadline:
    if child.poll() is not None:
        print(log.read_text(
            encoding="utf-8", errors="replace"
        )[-5000:])
        raise SystemExit(
            "服务启动退出；恢复14B可运行本目录的 restore14.sh"
        )
    try:
        served = request("/v1/models")["data"]
        assert any(
            m["id"] == target["alias"]
            and Path(m["root"]).resolve() == target["path"]
            for m in served
        )
        result = request("/v1/chat/completions", dict(
            model=target["alias"],
            messages=[dict(role="user", content="Reply OK.")],
            temperature=0, max_tokens=1
        ))
        assert result.get("choices")
        print(
            "模型就绪，实际生成检查通过:",
            target["alias"], flush=True
        )
        break
    except Exception:
        if time.monotonic() - last_status >= 10:
            print("等待模型加载，可查看日志:", log, flush=True)
            last_status = time.monotonic()
        time.sleep(3)
else:
    raise SystemExit(
        f"等待就绪超时，请查看日志 {log}；不要重复启动服务"
    )
