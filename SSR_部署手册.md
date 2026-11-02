# SSR 环境部署手册

适用代码版本：`SSR` @ `1003f62`（SIGIR 2026 官方仓库）。服务器：`A6000x1-3-168` / `A6000x1-7-168`。

**按顺序执行，每节末尾「本节验收」全部通过后再进入下一节。** 每段命令后面都有一张作用表，逐行说明这条命令改动了什么。

---

## 0. 环境约定与全流程总览

### 0.1 环境约定

| 项 | 值 |
|---|---|
| 服务器别名 | `A6000x1-3-168`（跑 vLLM）、`A6000x1-7-168`（备用） |
| 登录 | `ssh -p 38369 cs_tcci_zengdexin@210.40.16.205`（3-168）；`-p 32133` 是 7-168 |
| GPU | NVIDIA RTX A6000 48 GB（驱动 550.54.15） |
| HOME | `/root/user/cs_tcci_zengdexin`（共享盘 `172.30.20.24:6789:/ltaidc`） |
| 代码 | `~/SSR`（仓库根 = 运行根，`src/`、`datasets/`、`models/`、`artifacts/`、`logs/` 同层） |
| 数据 | `~/SSR/datasets/MultiTQ`（`kg/`、`questions/`、`prompt/`） |
| 模型 | `~/SSR/models/bge-m3`、`~/SSR/models/qwen14b-awq` |
| 产物 | `~/SSR/artifacts`（索引 + 缓存）、`~/SSR/logs` |
| conda 环境 | `ssr`（跑实验）、`vllm`（起推理服务），两者 torch 版本不同，禁止混用 |

### 0.2 全流程一览

| 节 | 做什么 | 关键产物 | 首次耗时 |
|---|---|---|---|
| 1 | 登录、确认共享盘、建运行根 | `~/SSR`（空目录） | 2 分钟 |
| 2 | 装 Miniconda + `ssr` 环境 | `~/miniconda3/envs/ssr` | 15 分钟 |
| 3 | 拉代码进运行根并钉到 `1003f62` | `~/SSR/{src,.git}` + `.git/info/exclude` | 1 分钟 |
| 4 | 准备 MultiTQ 数据 | `datasets/MultiTQ/{kg,questions,prompt}` | 5 分钟 |
| 5 | 下载 BGE-M3 | `models/bge-m3`（约 2.2 GB） | 5 分钟 |
| 6 | 三处代码改动（必须） | `gp_prompt.txt`、`qa_prompt.txt` 各追加 3 行；`graph_retriever.py` 三方法替换 | 6 分钟 |
| 7 | 建 `vllm` 环境 | `envs/vllm` | 15 分钟 |
| 8 | 下载 Qwen2.5-14B-AWQ | `models/qwen14b-awq`（9.4 GB） | 10–40 分钟 |
| 9 | 启动 vLLM 并体检 | 显存约 39980 MiB（跑实验时约 45996）、`/v1/models` 有 `qwen14b` | 5 分钟 |
| 10 | 冒烟 1 题 → 200 题 → 全量 | `artifacts/{索引,llm_cache.jsonl}`、`logs/*.log` | 1 题 1 分钟；200 题 5 分钟；全量约 20 小时 |
| 11 | 出错时查表 | — | — |
| 12 | tmux 速查、换机器 | — | — |

### 0.3 三条铁律

1. **所有实验命令都在 `~/SSR` 下执行。** `src/config.py` 里全是 `./datasets`、`./artifacts` 这样的相对路径，换个目录就找不到数据。
2. **实验一律走启动器，别自己 export。** 走 `bash ~/SSR/tools/run_ssr.sh <mode>`（第 10.1 节）时，三个 `OPENAI_*` 变量由启动器设好；只有你手工敲 `python src/run_pipeline.py` 时才需要自己 export（`config.py` 只从环境变量读，缺 `OPENAI_BASE_URL` 就会去连 `api.openai.com` 并报 401）。
3. **两台机器共写同一个 `artifacts/`。** 同一时刻只在一台机器上跑实验，否则索引与 `llm_cache.jsonl` 会互相覆盖。

### 0.4 最容易踩的三个坑

| 坑 | 症状 | 去哪查 |
|---|---|---|
| 用 `python -m` 模块方式启动 | `No module named 'src'` | 第 11 节第 1 条 |
| 没做第 6 节，模型只输出散文 | pattern 恒为 `("?","?","?")`、准确率固定在 0.2 附近 | 第 11 节第 3、8 条 |
| 没清 `llm_cache.jsonl` | Step 2 进度条显示几千 it/s（假命中缓存，不是加速） | 启动器已自动清，见第 10.2 节 |

### 本节验收

- `df -h ~/SSR` 显示 `172.30.20.24:/ltaidc`；
- `~/SSR` 下已有 `datasets/ models/ artifacts/ logs/`；
- 你能背出三条铁律（尤其"在 `~/SSR` 下执行"）。

---

## 1. 登录与目录准备

### 1.1 登录

```bash
ssh -p 38369 cs_tcci_zengdexin@210.40.16.205
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh -p 38369 ...` | 登录 3-168（跑 vLLM 的那台）；7-168 把端口换成 `32133` |

如果本机 `~/.ssh/config` 配了别名，可以直接 `ssh A6000x1-3-168`。

### 1.2 确认共享盘与环境

```bash
df -h ~/SSR
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `df -h ~/SSR` | 确认 `~/SSR` 挂在 NFS `172.30.20.24:6789:/ltaidc` 上（两台机共享同一份） |
| 2 | `nvidia-smi --query-gpu=...` | 确认看到 RTX A6000 48 GB、驱动 550.54.15、当前显存占用 |

### 1.3 建运行根

**所有东西都放在一个目录 `~/SSR` 里**：代码是 `~/SSR/src`，数据和模型在 `~/SSR/datasets`、`~/SSR/models`，产物和日志在 `~/SSR/artifacts`、`~/SSR/logs`。`~/SSR` 本身就是仓库根，也是每次运行的工作目录，代码、数据、模型、产物、日志都在这一个目录下面。

```bash
mkdir -p ~/SSR
ls -ld ~/SSR
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `mkdir -p ~/SSR` | 建运行根；其余子目录等到第 3、4 节（代码克隆进这个目录）和第 4.2 节（数据/产物/日志）再建，避免克隆时目录非空 |
| 2 | `ls -ld ~/SSR` | 确认存在、属主是你 |

### 1.4 确认可写

```bash
touch ~/SSR/.w && rm ~/SSR/.w && echo writable
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `touch ... && rm ...` | 往 `~/SSR` 写一个临时文件再删，验证 NFS 有写权限 |

### 本节验收

- `df -h ~/SSR` 出现 `172.30.20.24:/ltaidc`；
- `~/SSR` 存在且为空目录（接手现成机器时直接有内容，跳过第 3 节的克隆）；
- 最后一行打印 `writable`。

---

## 2. Miniconda 与 `ssr` 环境

### 2.1 下载并安装 Miniconda

```bash
cd ~
curl -fsSLO https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda --version
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd ~` | 回到 HOME，安装包落在 `$HOME` |
| 2 | `curl -fsSLO ...Miniconda3-latest-Linux-x86_64.sh` | 从清华镜像下载安装脚本（无 sudo 权限时唯一可行路径） |
| 3 | `bash ... -b -p ~/miniconda3` | `-b` 批处理静默安装，`-p` 指定装到 HOME 下 |
| 4 | `source ~/miniconda3/etc/profile.d/conda.sh` | 让当前 shell 认识 `conda`（不改 `~/.bashrc`） |
| 5 | `conda --version` | 验证 conda 可用 |

### 2.2 写清华源

```bash
cat > ~/.condarc <<'EOF'
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
channels:
  - defaults
show_channel_urls: true
EOF
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cat > ~/.condarc <<'EOF' ... EOF` | 写入清华源配置，后续 conda 安装走内网/镜像，避免境外源超时 |

### 2.3 创建 `ssr` 环境并装依赖

```bash
conda create -n ssr python=3.10 -y
conda activate ssr
pip install torch==2.4.1 -i https://mirrors.aliyun.com/pypi/simple
pip install faiss-cpu==1.8.0.post1 transformers==4.44.2 "tokenizers<0.20" sentencepiece "openai>=1.40,<2" tqdm "huggingface_hub>=0.33,<1.0" protobuf numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `conda create -n ssr python=3.10 -y` | 建实验环境（3.10 与 torch 2.4.1 / vLLM 0.8.5 都兼容） |
| 2 | `conda activate ssr` | 进入该环境，之后 `python` 都指向它 |
| 3 | `pip install torch==2.4.1 ...` | PyPI 上的 2.4.1 wheel 自带 cu121，匹配驱动 550 |
| 4 | `pip install faiss-cpu==1.8.0.post1 ...` | 装检索与 LLM 调用依赖：`faiss` 建索引、`transformers` 加载 BGE-M3、`openai` 打 vLLM、`numpy` 锁 1.26 以匹配 faiss 1.8 的 ABI；`huggingface_hub` 锁在 **1.0 以下**，因为第 5 节的 `huggingface-cli` 在 1.0 被移除（服务器上实装 0.36.2，`huggingface-cli download` 可用） |

### 2.4 验证

```bash
python -c "import torch, faiss, numpy, transformers, huggingface_hub as h; print(torch.__version__, torch.cuda.is_available(), faiss.__version__, numpy.__version__, transformers.__version__, h.__version__)"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python -c "import ..."` | 一次性导入五个关键库并打印版本，确认 CUDA 可用、hub 没越过 1.0 |

期望输出：`2.4.1+cu121 True 1.8.0 1.26.4 4.44.2 0.36.2`（`huggingface_hub` 只要在 `0.33`–`0.36` 之间即可，末位数字随安装日期变化）。

### 本节验收

- `conda env list` 里有 `ssr`；
- 上面那行输出里 `True` 和 `1.26.4` 必须对上，`huggingface_hub` 小于 `1.0`；
- `huggingface-cli --help` 第一行是 `usage: huggingface-cli <command> [<args>]`（第 5 节要用它下模型）。

---

## 3. 拉取代码

仓库直接克隆到运行根 `~/SSR`：`~/SSR/src/run_pipeline.py` 就是入口，数据和模型稍后落在同一个目录下。

克隆的是**改造后的仓库** `2200170080/SSR`：它在官方提交 `1003f62` 之上已经包含第 6 节的三处改动、启动器 `run_ssr.sh`、自动同步脚本 `tools/autosync.sh`，以及随仓库分发的 `datasets/MultiTQ/prompt/` 提示词。官方原版留作参照，需要时加一个 `upstream` remote 就能 `git diff`。

```bash
git clone git@github.com:2200170080/SSR.git ~/SSR
git -C ~/SSR log -1 --format='%h %s'
git -C ~/SSR remote -v
ls ~/SSR/src ~/SSR/datasets/MultiTQ/prompt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git clone git@github.com:2200170080/SSR.git ~/SSR` | 克隆自己的仓库到运行根（含三处改动、启动器、提示词）；没配 SSH key 就换 `https://github.com/2200170080/SSR.git` |
| 2 | `git -C ~/SSR log -1 --format='%h %s'` | 确认 HEAD 落在最新提交（`chore:` 或 `auto:` 开头），不是停在 `1003f62` |
| 3 | `git -C ~/SSR remote -v` | `origin` 指向 `2200170080/SSR`，这就是你要推的仓库 |
| 4 | `ls ~/SSR/src ~/SSR/datasets/MultiTQ/prompt` | 代码与提示词都在：`config.py`、`run_pipeline.py`、`retriever/`、`utils/`，以及 `gp_prompt.txt`、`qa_prompt.txt` |

想和官方原版对照时：

```bash
git -C ~/SSR remote add upstream https://github.com/zhangli-coding/SSR.git
git -C ~/SSR fetch upstream
git -C ~/SSR diff upstream/main --stat
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git remote add upstream ...` | 只加只读参照，不参与推送 |
| 2 | `git fetch upstream` | 拉官方分支到本地引用 |
| 3 | `git diff upstream/main --stat` | 看我们改了哪些文件（授权补丁、启动器、prompt、忽略规则） |

数据、模型、产物、日志都住在仓库目录里面。仓库的跟踪范围是**除了大二进制以外全部纳入版本管理**：`datasets/`、`logs/`、`patches/`、`*.npy`、`tools/` 都进 git；只有下面这几样被挡住——模型权重（14 GB）、两个检索索引（1.8 GB + 115 MB）、全量跑完后约 700 MB 的 `llm_cache.jsonl`、以及上游数据集仓库 `MultiTQ/`（自带 `.git`，一条 `git clone` 可重建）。GitHub 单文件上限 100 MB，仓库超过 1 GB 就不实用了。

这一步**只做一次**（`.gitignore` 里已有同样一份，本机这份是兜底）：

```bash
cat >> ~/SSR/.git/info/exclude <<'EOF'

# never commit: model weights, built indexes, response cache, upstream dataset repo
/models/
/artifacts/multitq_text_index.pkl
/artifacts/multitq_graph_retriever.pkl
/artifacts/llm_cache.jsonl
/MultiTQ/
/.hf_cache/
EOF
git -C ~/SSR status --porcelain
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cat >> ~/SSR/.git/info/exclude <<'EOF' … EOF` | 把模型权重、检索索引、响应缓存、上游仓库登记为"本地运行数据"，只影响本机；`datasets/`、`logs/`、`patches/`、`*.npy` 不在此列——它们是要同步的 |
| 2 | `git -C ~/SSR status --porcelain` | 应只剩 ` M src/retriever/graph_retriever.py`（第 6.3 节的补丁）；若列出 `models/`、`artifacts/*.pkl`，说明上面那段没写进去 |

### 本节验收

- `git -C ~/SSR log --oneline | tail -1` 能看到 `1003f62`（说明历史接得上官方版本），`git log -1` 是最新提交；
- `git -C ~/SSR status --porcelain` **无输出**（三处改动都已提交在本仓库里，不是你本地改的）；只有你自己动过文件才会列出来；
- `git -C ~/SSR ls-files | grep -c '^datasets/'` 应为 **6**（`kg/` 三个 json+txt、`questions/test.json`、`prompt/` 两个 txt），说明数据确实在版本管理里。

> 如果你接手的是**别人已经部署好的机器**，跳过 `git clone`，只看两点：`~/SSR/src/run_pipeline.py` 在不在、上面那段 exclude 有没有。`status` 里出现 ` M src/retriever/graph_retriever.py` 是正常的（第 6.3 节的容错补丁），判断环境就绪走第 6.4 节。

---

## 4. 准备 MultiTQ 数据

### 4.1 下载并解压

```bash
cd ~/SSR
git clone https://github.com/czy1999/MultiTQ.git
cd ~/SSR/MultiTQ/data && unzip -o Dataset.zip
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd ~/SSR` | 数据仓库克隆到 `~/SSR/MultiTQ` |
| 2 | `git clone https://github.com/czy1999/MultiTQ.git` | 拉 MultiTQ 数据仓库（`data/Dataset.zip` 内含 KG 与问题集） |
| 3 | `cd ~/SSR/MultiTQ/data && unzip -o Dataset.zip` | 解压数据，得到 `MultiTQ/kg`、`MultiTQ/questions` |

### 4.2 摆成 SSR 期望的目录

```bash
mkdir -p ~/SSR/{datasets/MultiTQ/prompt,artifacts,logs,patches}
cp -r ~/SSR/MultiTQ/data/MultiTQ/kg ~/SSR/datasets/MultiTQ/
cp -r ~/SSR/MultiTQ/data/MultiTQ/questions ~/SSR/datasets/MultiTQ/
ls ~/SSR/datasets/MultiTQ/prompt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `mkdir -p ~/SSR/{datasets/MultiTQ/prompt,artifacts,logs,patches}` | 建数据目标目录，同时把产物、日志、补丁备份三个目录一起建出来（第 9、10 节要用） |
| 2 | `cp -r .../kg ~/SSR/datasets/MultiTQ/` | 拷贝三元组与 id 映射（`full.txt`、`entity2id.json`、`relation2id.json`） |
| 3 | `cp -r .../questions ~/SSR/datasets/MultiTQ/` | 拷贝问题集（`test.json` 等） |
| 4 | `ls ~/SSR/datasets/MultiTQ/prompt` | 提示词随仓库一起下来（`datasets/MultiTQ/prompt/` 是 git 跟踪的），确认 `gp_prompt.txt`、`qa_prompt.txt` 都在；`config.py` 读的就是这一处，全流程只有这一份 |

拷完之后 `~/SSR/MultiTQ/`（clone 下来的那个）就**没用了**：它是下载源，里面是 `Dataset.zip`、`MultiQA` 代码，还有 SSR 根本不读的 `train.json`（131 MB）/ `dev.json`（20 MB），整目录 479 MB，代码里没有任何一处引用它。确认 `datasets/MultiTQ/` 拷完、第 4.3 节校验通过后可以删掉：

```bash
rm -rf ~/SSR/MultiTQ
du -sh ~/SSR
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `rm -rf ~/SSR/MultiTQ` | 删掉下载源目录（`datasets/` 里的运行时数据不受影响）；要留离线备份就只删 `rm -rf ~/SSR/MultiTQ/data/MultiTQ ~/SSR/MultiTQ/MultiQA`，保留 40 MB 的 `Dataset.zip` |
| 2 | `du -sh ~/SSR` | 确认总占用降下来（含模型前约 700 MB 级） |

### 4.3 规模与位置校验

```bash
wc -l ~/SSR/datasets/MultiTQ/kg/full.txt
python -c "import json; print(len(json.load(open('/root/user/cs_tcci_zengdexin/SSR/datasets/MultiTQ/questions/test.json'))))"
ls ~/SSR/datasets/MultiTQ/kg/entity2id.json ~/SSR/datasets/MultiTQ/kg/relation2id.json
ls ~/SSR/datasets/MultiTQ/prompt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `wc -l .../kg/full.txt` | 期望 `461329` 行；少很多说明 `Dataset.zip` 没解压完整 |
| 2 | `python -c "print(len(...test.json))"` | 期望 `54584` 题 |
| 3 | `ls .../kg/entity2id.json .../kg/relation2id.json` | `config.py` 要求这两个文件在 `kg/` 下；若解压后在外层，用 `mv` 挪进 `kg/` |
| 4 | `ls ~/SSR/datasets/MultiTQ/prompt` | 应有 `gp_prompt.txt` 等模板 |

若第 3 条报 `No such file`：

```bash
mv ~/SSR/datasets/MultiTQ/entity2id.json ~/SSR/datasets/MultiTQ/kg/ 2>/dev/null
mv ~/SSR/datasets/MultiTQ/relation2id.json ~/SSR/datasets/MultiTQ/kg/ 2>/dev/null
ls ~/SSR/datasets/MultiTQ/kg/*.json
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `mv .../entity2id.json .../kg/` | 把外层多出来的 id 映射挪进 `kg/`（已在里面时静默失败） |
| 2 | `mv .../relation2id.json .../kg/` | 同上，关系映射 |
| 3 | `ls .../kg/*.json` | 确认两个文件就位 |

### 本节验收

- `full.txt` 为 461329 行；
- `test.json` 为 54584 题；
- `kg/` 下有 `entity2id.json`、`relation2id.json`；
- `prompt/` 下有 `gp_prompt.txt`。

---

## 5. 下载 BGE-M3

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/SSR/.hf_cache
huggingface-cli download BAAI/bge-m3 --local-dir ~/SSR/models/bge-m3 --exclude "imgs/*" "onnx/*"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `export HF_ENDPOINT=https://hf-mirror.com` | 走国内镜像，避免直连 huggingface.co 超时 |
| 2 | `export HF_HOME=~/SSR/.hf_cache` | 把下载缓存放到共享盘，换机器不必重下 |
| 3 | `huggingface-cli download BAAI/bge-m3 --local-dir ...` | 下模型到 `~/SSR/models/bge-m3`；`--exclude` 掉示例图和 onnx 目录，避免无谓 403 中断导致缺 `tokenizer.json` |

### 5.1 校验

```bash
du -sh ~/SSR/models/bge-m3
python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/root/user/cs_tcci_zengdexin/SSR/models/bge-m3")
print(type(t).__name__)
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `du -sh ~/SSR/models/bge-m3` | 期望约 **2.2 GB**（只有 `pytorch_model.bin` 时）；若目录里还躺着 `onnx/`，会显示约 **4.3 GB**——见下方说明 |
| 2 | `python - <<'PY' ... PY` | 加载分词器，期望打印 `XLMRobertaTokenizerFast`；若是慢分词器 `XLMRobertaTokenizer`，说明 `tokenizer.json` 没下全，重跑下载 |

**关于 `onnx/` 与体积**：`--exclude "onnx/*"` 并不总能拦住——实测服务器上 `onnx/` 仍然下全了（2.2 GB），所以 `du -sh` 报 4.3 GB。运行时只加载 `pytorch_model.bin`（2.27 GB），ONNX 目录纯属占地方，想省空间可以删：

```bash
rm -rf ~/SSR/models/bge-m3/onnx
du -sh ~/SSR/models/bge-m3
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `rm -rf ~/SSR/models/bge-m3/onnx` | 删掉没用的 ONNX 导出（代码只读 `pytorch_model.bin`），省 2.2 GB |
| 2 | `du -sh ~/SSR/models/bge-m3` | 删完应回到约 2.2 GB |

### 本节验收

- `pytorch_model.bin` 存在且约 2.27 GB（`ls -l ~/SSR/models/bge-m3/pytorch_model.bin`）；
- 分词器类型为 `XLMRobertaTokenizerFast`。

---

## 6. 代码改动（必须，共三处）

> 这三处不是"优化"，是**跑得通的前提**。克隆本仓库时它们**已经提交在代码里**——所以第 3 节验收里的 `git status --porcelain` 是干净的。只有两种情况需要往下走：①你是从官方 `1003f62` 原版起步；②不小心 `git checkout -- .` 还原过。不确定就先做 6.4 的三条 `grep`，哪条不对补哪节。
>
> 6.1 / 6.2 / 6.3 三段都是**幂等**的，重复执行只打印 `skip:` / `already patched`，不会重复追加。改动前建议先看 `git -C ~/SSR status`，出问题用 `git -C ~/SSR checkout -- .` 回滚（6.1/6.2 改的提示词就在 `datasets/MultiTQ/prompt/` 下，是 git 跟踪的文件，回滚会连它们一起还原）。

### 6.1 让模型只输出两行（改提示词，追加 3 行）

改的是 `datasets/MultiTQ/prompt/gp_prompt.txt`——`config.py` 只读这一份，它也在 git 里（改完会被自动同步到你的仓库）：

```bash
python - <<'PY'
from pathlib import Path

SUFFIX = (
    "\nDo not explain your reasoning. Output ONLY these two lines and nothing else:\n"
    "<fact> head entity; relation; tail entity </fact>\n"
    "<time> start date; end date </time>\n"
)
p = Path("/root/user/cs_tcci_zengdexin/SSR/datasets/MultiTQ/prompt/gp_prompt.txt")
text = p.read_text()
if "Output ONLY these two lines" in text:
    print("skip:", p)
else:
    p.write_text(text + SUFFIX)
    print("patched:", p)
PY
tail -4 ~/SSR/datasets/MultiTQ/prompt/gp_prompt.txt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python - <<'PY' ... PY` | 给 `gp_prompt.txt` 末尾追加三行格式约束（禁止解释 + `<fact>` 行 + `<time>` 行） |
| 2 | `if "Output ONLY these two lines" in text: print("skip")` | 幂等保护：已追加过就跳过，重复执行不会叠加 |
| 3 | `p.write_text(text + SUFFIX)` | 文件末尾追加：禁止解释 + `<fact>` 一行 + `<time>` 一行 |
| 4 | `tail -4 ~/SSR/datasets/MultiTQ/prompt/gp_prompt.txt` | 确认最后几行就是格式约束 |

**为什么必须改**：模型原本只写一段散文（实测 4.7 s、256 token、无任何标签），`query_process.py` 抽不到 `<fact>`/`<time>`，pattern 恒为 `("?","?","?")`，图检索永远为空、退回文本 top-50，准确率就卡在 0.2 附近。加这三行后实测 0.8 s、43 token，输出恰为：

```
<fact> ?; Make a visit; China </fact>
<time> 2013-05-08; 2013-05-08 </time>
```

### 6.2 让回答必定带答案句（改 `qa_prompt.txt`，追加 3 行）

`query_process.py:68` 的 `get_qa_result` 是这样抽答案的：响应里没有 `So the answer is` 就直接 `return "ERROR"`，这道题必判错。Qwen2.5-14B 约 **1/4** 的题会写成 "The provided historical facts do not contain any information…" 这类散文，白丢分。下面在 `qa_prompt.txt` 末尾追加三行：

```bash
python - <<'PY'
from pathlib import Path

SUFFIX = (
    "\nYou must end your response with exactly one final line of the form:\n"
    'So the answer is: ["entity1", "entity2"]\n'
    "Do not write anything after that line.\n"
)
p = Path("/root/user/cs_tcci_zengdexin/SSR/datasets/MultiTQ/prompt/qa_prompt.txt")
text = p.read_text()
if "one final line of the form" in text:
    print("skip:", p)
else:
    p.write_text(text + SUFFIX)
    print("patched:", p)
PY
tail -4 ~/SSR/datasets/MultiTQ/prompt/qa_prompt.txt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python - <<'PY' ... PY` | 给 `qa_prompt.txt` 末尾追加三行：必须以 `So the answer is: [...]` 收尾 |
| 2 | `if "one final line of the form" in text: print("skip")` | 幂等保护：已追加过就跳过，重复执行不会叠加 |
| 3 | `tail -4 ~/SSR/datasets/MultiTQ/prompt/qa_prompt.txt` | 确认最后三行就是格式约束（`Answer:` 之后） |

**实测效果**（同一批 20 条真实 QA prompt、并发 8）：加之前 **15/20** 的输出带 `So the answer is`，加之后 **20/20**；总耗时 7.3 s 不变、生成 token 1,951 → 1,957。也就是说，这 5 道本来会被判 `ERROR` 的题被救回来了，代价为零。

> 这一处只管"能不能被判分"，不管"答对没答对"。它和第 6.1 节的 `<fact>`/`<time>` 约束是两件事：6.1 修的是**检索**（图模式抽不出来），这里修的是**判分**（答案句抽不出来）。

### 6.3 让日期解析不崩（改 `graph_retriever.py` 三个方法）

`_parse_time` 原版对无法解析的串直接 `raise`，模型一旦写出 `start_date` 这类占位词，整个 5.4 万题的批次会中途死掉。下面把 `_parse_time`、`_sort_time_key`、`_fact_matches_time` 三个方法整段替换为容错版本（幂等，可重复执行）：

```bash
cd ~/SSR
python - <<'PY'
from pathlib import Path

p = Path("src/retriever/graph_retriever.py")
s = p.read_text()

NEW = '''    @staticmethod
    def _parse_time(raw_time: str) -> datetime:
        raw_time = raw_time.strip()
        if raw_time == "?":
            return datetime.max
        if raw_time.isdigit():
            year = int(raw_time)
            if year <= 0:
                year = 1
                # raise ValueError(f"Year must be >= 0: {raw_time}")
            return datetime(year, 1, 1, 0, 0, 0)
        if "-" in raw_time and len(raw_time.split("-")) == 3 and " " not in raw_time:
            try:
                y, m, d = raw_time.split("-")
                if int(y) <= 0:
                    y = 1
                return datetime(int(y), int(m), int(d), 0, 0, 0)
            except ValueError:
                return None
        try:
            return datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _sort_time_key(self, fact) -> datetime:
        fact_time = fact[-1]
        if isinstance(fact_time, tuple):
            start_time = self._parse_time(fact_time[0])
            end_time = self._parse_time(fact_time[1])
        else:
            start_time = self._parse_time(fact_time)
            end_time = start_time
        known = [t for t in (start_time, end_time) if t is not None]
        return min(known) if known else datetime.max

    def _fact_matches_time(self, fact, start_time: str, end_time: str) -> bool:
        fact_time = fact[-1]
        if isinstance(fact_time, tuple):
            fact_start = self._parse_time(fact_time[0])
            fact_end = self._parse_time(fact_time[1])
        else:
            fact_start = self._parse_time(fact_time)
            fact_end = fact_start

        if fact_start is None or fact_end is None:
            return True

        if start_time != "?":
            query_start = self._parse_time(start_time)
            if query_start is not None and fact_end < query_start:
                return False
        if end_time != "?":
            query_end = self._parse_time(end_time)
            if query_end is not None and fact_start > query_end:
                return False
        return True

'''

if 'try:\n            y, m, d = raw_time.split("-")' in s:
    print("already patched")
else:
    start = s.index("    @staticmethod\n    def _parse_time")
    end = s.index("    def get_facts(")
    p.write_text(s[:start] + NEW + s[end:])
    print("patched: src/retriever/graph_retriever.py")
PY
python -m py_compile src/retriever/graph_retriever.py && echo "syntax OK"
grep -c "except ValueError" src/retriever/graph_retriever.py
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd ~/SSR` | 进入仓库根，后面用 `src/...` 相对路径 |
| 2 | `python - <<'PY' ... PY` | 用脚本按"起始锚点 → `def get_facts(`"整段替换三个方法；含幂等判断，已打过就只打印 `already patched` |
| 3 | `python -m py_compile src/retriever/graph_retriever.py` | 语法自检——粘贴代码最容易出的错是缩进，这里能立刻发现 |
| 4 | `grep -c "except ValueError" ...` | 期望输出 `2`（两处 `try/except` 都到位） |

**这段改动的语义**：原本能解析的路径（`?`、纯年份、`yyyy-mm-dd`、`yyyy-mm-dd HH:MM:SS`）行为不变；只有原本会抛异常的那条路径变成"返回 `None`"，并在比对时按"该侧无约束"处理。所以不会影响正常结果，只是不再整批崩掉。

⚠️ **`try` 必须从 `y, m, d = raw_time.split("-")` 之前开始**。模型会写出 `?2011` 这种串，若 `try` 只包住 `datetime(...)`，异常会先在 `int(y)` 处抛出（报 `invalid literal for int() with base 10: '?2011'`），5.4 万题的批次照样中途死掉。

### 6.4 接手现成环境时：先核对三处改动是否已落地

从零部署按 6.1→6.3 走即可；如果你是接手一台**已经在跑**的机器，别急着重新打补丁（补丁是幂等的，但"以为没打"最费时间），先量一下现状：

```bash
grep -c '<fact>' ~/SSR/datasets/MultiTQ/prompt/gp_prompt.txt
grep -c 'one final line of the form' ~/SSR/datasets/MultiTQ/prompt/qa_prompt.txt
grep -c "except ValueError" ~/SSR/src/retriever/graph_retriever.py
sed -n '113,122p' ~/SSR/src/retriever/graph_retriever.py
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `grep -c '<fact>' .../gp_prompt.txt` | 数生效版 gp 提示词里 `<fact>` 的行数；6.1 落地后应为 **5** |
| 2 | `grep -c 'one final line of the form' .../qa_prompt.txt` | 数生效版 qa 提示词里的答案句约束；6.2 落地后应为 **1** |
| 3 | `grep -c "except ValueError" .../graph_retriever.py` | 6.3 落地后应为 **2**（两处 `try/except`） |
| 4 | `sed -n '113,122p' .../graph_retriever.py` | 眼查 `try:` 是否在 `y, m, d = raw_time.split("-")` **之前** |

对照表（2026-11-02 在 `A6000x1-3-168` 上实测，可直接当"已部署"的样子）：

| 检查项 | 位置 | 已打补丁的样子 |
|---|---|---|
| `gp_prompt.txt` | `~/SSR/datasets/MultiTQ/prompt/`（`config.py` 唯一读的那份，git 跟踪） | `fact_tag=5`，55 行 |
| `qa_prompt.txt` | 同上 | `answer_line=1`，58 行 |
| `graph_retriever.py` | `~/SSR/src/retriever/` | `except ValueError count=2`，且 `try:` 在 `y, m, d = split("-")` 之前 |

判读规则：

- **任一项对不上** → 回到 6.1/6.2/6.3 对应小节重打一遍（三步都是幂等的，重复执行不会叠加），然后清缓存（10.2）再跑实验。
- 提示词现在只有 `datasets/MultiTQ/prompt/` 这一份，且被 git 跟踪：改了会出现在 `git status` 里、也会被自动同步推到你的仓库；`git checkout -- .` 会把它一起还原。

### 本节验收

- `tail -4 ~/SSR/datasets/MultiTQ/prompt/gp_prompt.txt` 能看到 `<fact>` 与 `<time>` 两行格式（`grep -c '<fact>'` 为 5）；
- `tail -4 ~/SSR/datasets/MultiTQ/prompt/qa_prompt.txt` 末三行是 `So the answer is` 约束（`grep -c 'one final line of the form'` 为 1）；
- `py_compile` 打印 `syntax OK`；
- `grep -c "except ValueError"` 输出 `2`；
- `sed -n '113,122p' src/retriever/graph_retriever.py` 里 `try:` 出现在 `y, m, d = raw_time.split("-")` **之前**（在 `if int(y) <= 0` 之前）；
- `python -c "import ast,sys; ast.parse(open('src/retriever/graph_retriever.py').read())"` 不报错（可选二次确认）。

---

## 7. 创建 `vllm` 环境

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install "vllm==0.8.5" -i https://mirrors.aliyun.com/pypi/simple
pip install "transformers==4.51.3" "huggingface_hub<1.0" -i https://mirrors.aliyun.com/pypi/simple
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `conda create -n vllm python=3.10 -y` | 推理环境独立，避免与 `ssr` 的 torch 2.4.1 冲突 |
| 2 | `conda activate vllm` | 进入推理环境 |
| 3 | `pip install "vllm==0.8.5"` | vLLM 自带 torch 2.6.0+cu124（匹配驱动 550）；**不要装最新版** |
| 4 | `pip install "transformers==4.51.3" "huggingface_hub<1.0"` | 把 transformers 降到与 0.8.5 配套的版本，修复 `Qwen2Tokenizer has no attribute all_special_tokens_extended` |

### 7.1 验证

```bash
python - <<'PY'
import vllm, torch, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__,
      "| cu", torch.version.cuda, "| ok", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python - <<'PY' ... PY` | 在 `vllm` 环境里一次导入三者，打印 vLLM/torch/CUDA 版本、CUDA 是否可用、transformers 版本 |

期望：`vllm 0.8.5 | torch 2.6.0+cu124 | cu 12.4 | ok True` 与 `transformers 4.51.3`。

### 本节验收

- vLLM 为 `0.8.5`、transformers 为 `4.51.3`、`torch.cuda.is_available()` 为 `True`。

---

## 8. 下载 Qwen2.5-14B-Instruct-AWQ

```bash
conda activate ssr
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
modelscope download --model Qwen/Qwen2.5-14B-Instruct-AWQ --local_dir ~/SSR/models/qwen14b-awq
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `conda activate ssr` | modelscope 装在哪个环境都行，这里就近用 `ssr` |
| 2 | `pip install modelscope` | 国内下载比 HF 稳、不需要 token |
| 3 | `modelscope download ... --local_dir ~/SSR/models/qwen14b-awq` | 下 AWQ 量化权重（约 9.4 GB）到共享盘 |

### 8.1 验证

```bash
du -sh ~/SSR/models/qwen14b-awq
ls ~/SSR/models/qwen14b-awq
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `du -sh ...` | 期望约 9.4 G |
| 2 | `ls ...` | 应有 `config.json`、`model-0000X-of-00003.safetensors`、`tokenizer.json` |

### 本节验收

- 目录约 9.4 G；
- 三个关键文件齐全（缺 `tokenizer.json` 会在起服务时报分词器错误）。

---

## 9. 启动 vLLM 服务

### 9.1 起服务（nohup 守护，推荐）

```bash
nvidia-smi --query-gpu=memory.used --format=csv
mkdir -p ~/SSR/logs
nohup ~/miniconda3/envs/vllm/bin/vllm serve ~/SSR/models/qwen14b-awq \
  --served-model-name qwen14b --port 8000 \
  --max-model-len 8192 --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 --disable-log-requests \
  > ~/SSR/logs/vllm-3-168.log 2>&1 &
disown
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `nvidia-smi --query-gpu=memory.used --format=csv` | 起服务前确认卡是空的（占用应接近 0 MiB） |
| 2 | `mkdir -p ~/SSR/logs` | 日志目录 |
| 3 | `nohup ~/miniconda3/envs/vllm/bin/vllm serve ... &` | 用**绝对路径二进制**起 OpenAI 兼容服务，不依赖 `conda activate`（和 `ssr` 环境那边同一个道理）；`--served-model-name qwen14b` 决定客户端里的 `model` 名；`--max-model-len 8192` 覆盖本项目 prompt；`--max-num-seqs 64` 支持并发 8 的实验（别往上调，见 10.6）；`--gpu-memory-utilization 0.80` 留出约 2 GB 余量给 BGE-M3；`--disable-log-requests` 减少日志噪音（代价：日志里看不到请求记录） |
| 4 | `disown` | 把任务从 shell 作业表摘掉，断开 ssh 不会收到 SIGHUP |

**为什么用 nohup 而不是 tmux**：这样起出来的进程 `PPID=1`、`TTY=?`，已经完全脱离终端；tmux 只是另一层壳，关掉终端后反而多一层依赖。tmux 留给第 10 节的前台实验进程。

### 9.2 起服务的备选写法（tmux）

```bash
tmux new -d -s vllm -c ~/SSR "~/miniconda3/envs/vllm/bin/vllm serve ~/SSR/models/qwen14b-awq --served-model-name qwen14b --port 8000 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.80 --disable-log-requests 2>&1 | tee ~/SSR/logs/vllm-tmux.log"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s vllm ...` | 在名为 `vllm` 的后台会话里起服务，日志同时进文件；需要交互调试时用这种 |

两种方式**只能选一种**，同一张卡不能起第二个 vLLM。

### 9.3 等就绪

```bash
timeout 480 bash -c 'until curl -sf http://127.0.0.1:8000/v1/models | grep -q qwen14b; do sleep 5; done' && echo "vLLM ready"
tail -20 ~/SSR/logs/vllm-3-168.log | grep -E "Application startup complete|error|Error"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `timeout 480 bash -c 'until curl -sf .../v1/models \| grep -q qwen14b; do sleep 5; done'` | 每 5 秒探一次，最多等 8 分钟；打印 `vLLM ready` 才说明真的起来了。冷启动实测约 **4–5 分钟**（9.4 GB 权重从共享盘读 + `torch.compile` 首次编译 100 秒），别用固定 `sleep 90` |
| 2 | `tail -20 ... \| grep -E "Application startup complete\|error\|Error"` | 日志里出现 `Application startup complete` 即成功；同时扫一眼有没有 `error` |

### 9.4 体检（三项同时成立才算过）

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen14b","messages":[{"role":"user","content":"只输出这一句：So the answer is hello"}],"temperature":0,"max_tokens":64}' | head -c 300
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `curl /v1/models` | 应看到 `"id":"qwen14b"`，说明服务与模型名对得上 |
| 2 | `curl /v1/chat/completions ...` | 发一次最小对话，确认真能生成（这是后面 `query_process.py` 依赖的接口） |
| 3 | `nvidia-smi --query-gpu=memory.used,utilization.gpu` | 期望显存约 39980 MiB（稳态实测 39980；启动过程中会先看到 39868 左右，浮动几百 MiB 属正常） |

**显存判据**：约 39980 MiB = 0.80 × 49140，这是 vLLM 给自己留的 KV cache 配额（Qwen2.5-14B-AWQ 权重 9.4 GB，其余给 64 路 × 8192 的并发），**不是本项目额外需求**；实验侧 BGE-M3 编码 46 万条事实的峰值只有约 6.4 GiB，`0.80` 的余量足够。

**跑到实验阶段会涨到约 46000 MiB**（2026-11-01 15:52 实测 45996 MiB）：vLLM 的约 39980 不动，多出来的是 `ssr` 环境里的 BGE-M3（fp32、约 2.3 GB 权重 + 激活）在同一张卡上做编码。49,140 MiB 的卡还剩约 3 GB，属正常，不要以为漏了进程。看到 `45000+` 就别再往上加并发或换更大的 `--max-model-len`。

### 本节验收

- `/v1/models` 返回 `qwen14b`，且响应里的 `root` 指向 `~/SSR/models/qwen14b-awq`（路径变了必须重启服务，见 9.1）；
- chat 接口有正常回复（含 `So the answer is`）；
- 显存约 39980 MiB，且只有一个 vLLM 进程（`pgrep -af "vllm serve" | wc -l` 为 1）；跑实验时涨到约 45996 MiB 属正常（见 10.6）。

---

## 10. 跑实验

> 本节所有实验命令都通过一个启动器 `~/SSR/tools/run_ssr.sh` 执行（10.1 装一次）。它内部用绝对路径 `~/miniconda3/envs/ssr/bin/python`、自己设三个 `OPENAI_*` 变量，所以**不需要 `(ssr)` 前缀，也不会因为 conda 没激活而静默失败**。下面的命令在任何 shell、任何 tmux 窗口里都一样能用。

### 10.1 一次性准备启动器（解决「没有 (ssr) 前缀」）

```bash
mkdir -p ~/SSR/tools
cat > ~/SSR/tools/run_ssr.sh <<'EOF'
#!/bin/bash
# run_ssr.sh -- SSR pipeline launcher that does NOT depend on `conda activate`.
# Usage: bash ~/SSR/tools/run_ssr.sh [check|smoke1|smoke20|smoke200|full]
# Every run clears artifacts/llm_cache.jsonl first; prefix SSR_KEEP_CACHE=1 to resume instead.
# All output goes to both the terminal (tmux pane) and logs/<tag>-<MMDD-HHMM>.log
set -u

PY="$HOME/miniconda3/envs/ssr/bin/python"
ROOT="$HOME/SSR"
MODEL="$ROOT/models/bge-m3"
CODE="src/run_pipeline.py"

[ -x "$PY" ] || { echo "ERROR: python not found: $PY"; exit 1; }
[ -d "$ROOT" ] || { echo "ERROR: run root not found: $ROOT"; exit 1; }
[ -f "$ROOT/$CODE" ] || { echo "ERROR: pipeline not found: $ROOT/$CODE"; exit 1; }

MODE="${1:-full}"

# FAISS/OpenMP: this node has 128 logical cpus and is shared (load ~150).
# Letting OpenMP grab all 128 threads makes flat-IP search ~14x SLOWER (89 s vs 6.2 s per batch).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

case "$MODE" in
  check)    ARGS=() ;;
  smoke1)   ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 1 --concurrency 1) ;;
  smoke20)  ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 20 --concurrency 8) ;;
  smoke200) ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 200 --concurrency 8 --random-seed 7) ;;
  full)     ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --concurrency 8) ;;
  *) echo "usage: bash ~/SSR/tools/run_ssr.sh [check|smoke1|smoke20|smoke200|full]"; exit 2 ;;
esac

echo "[run_ssr] mode=$MODE"
echo "[run_ssr] python=$PY  OMP_NUM_THREADS=$OMP_NUM_THREADS"
"$PY" -c 'import sys,torch,transformers;print("[run_ssr] python",sys.version.split()[0],"torch",torch.__version__,"cuda",torch.cuda.is_available(),"tf",transformers.__version__)'

if [ "$MODE" = "check" ]; then
  "$PY" -c 'import faiss;print("[run_ssr] faiss.omp_get_max_threads()=",faiss.omp_get_max_threads())'
  echo "[run_ssr] check OK (launcher works without conda activate)"
  exit 0
fi

cd "$ROOT" || { echo "ERROR: cd $ROOT failed"; exit 1; }
export PYTHONUNBUFFERED=1
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL_NAME="qwen14b"

# 每次实验前自动清缓存；断点续跑时 SSR_KEEP_CACHE=1 可跳过
if [ "${SSR_KEEP_CACHE:-0}" = "1" ]; then
  echo "[run_ssr] keep llm_cache.jsonl (resume mode: SSR_KEEP_CACHE=1)"
else
  rm -f "$ROOT/artifacts/llm_cache.jsonl"
  echo "[run_ssr] cleared artifacts/llm_cache.jsonl (旧缓存会让 Step 2 假加速)"
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${MODE}-$(date +%m%d-%H%M).log"
echo "[run_ssr] start=$(date '+%F %T')  log=$LOG"
"$PY" "$CODE" "${ARGS[@]}" --text-retriever-model-name "$MODEL" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[run_ssr] end=$(date '+%F %T')  exit=$RC"
exit "$RC"
EOF
chmod +x ~/SSR/tools/run_ssr.sh
bash ~/SSR/tools/run_ssr.sh check
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `mkdir -p ~/SSR/tools` | 建 `tools/` 目录（仓库里本来就有，防手滑删掉后写入失败） |
| 2 | `cat > ~/SSR/tools/run_ssr.sh <<'EOF' … EOF` | 写入启动器：绝对路径解释器 + 三个 `OPENAI_*` + **每次自动清缓存** + 按模式拼参数 + 输出同时进屏幕和 `logs/<模式>-<月日-时分>.log` |
| 3 | `chmod +x ~/SSR/tools/run_ssr.sh` | 加执行权限 |
| 4 | `bash ~/SSR/tools/run_ssr.sh check` | 只验解释器与路径、不跑实验；应输出 `python 3.10.22 torch 2.4.1+cu121 cuda True tf 4.44.2`、`faiss.omp_get_max_threads()= 8` 与 `check OK` |

模式一览：`check`（只自检）、`smoke1`（1 题）、`smoke20`（20 题）、`smoke200`（200 题、种子 7）、`full`（全量 54584 题）。**以后所有实验都用它，不再手写 `python src/run_pipeline.py`。**

**清缓存已经做进启动器**：每次跑实验，启动器会先 `rm -f artifacts/llm_cache.jsonl` 并打印一行 `[run_ssr] cleared artifacts/llm_cache.jsonl`，**你不用再自己敲 `rm -f`**。唯一例外见 10.2 的续跑写法。

### 10.2 关于缓存（启动器已自动清，你不用敲）

缓存键包含 prompt 全文，所以改过提示词后旧缓存既不会命中、也不会报错，只会让 Step 2 出现「几千 it/s」的假加速。因此启动器在每次实验开头自动清一次，日志里那行 `[run_ssr] cleared artifacts/llm_cache.jsonl` 就是证据；如果看到的是 `keep llm_cache.jsonl (resume mode...)`，说明你正处于续跑模式。

只有一种情况需要保留缓存：全量跑到一半断了要接着跑。

```bash
SSR_KEEP_CACHE=1 tmux new -d -s full -c ~/SSR 'bash ~/SSR/tools/run_ssr.sh full'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `SSR_KEEP_CACHE=1 tmux new …` | 续跑：让启动器跳过清缓存，复用已完成题目的 LLM 响应；改过 prompt 后旧键全部失配，等于从零开始，此时不要用这个前缀 |

### 10.3 单题先行（1 分钟）

```bash
bash ~/SSR/tools/run_ssr.sh smoke1
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `bash ~/SSR/tools/run_ssr.sh smoke1` | 跑 1 题（`--sample-size 1 --concurrency 1`）；`~/SSR`、三个 `OPENAI_*`、BGE-M3 路径、清缓存、日志文件全由启动器处理 |

**第一题阶段的两个必查点**：

```bash
grep -c '<fact>' ~/SSR/artifacts/llm_cache.jsonl
tail -3 $(ls -t ~/SSR/logs/smoke1-*.log | head -1)
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `grep -c '<fact>' .../llm_cache.jsonl` | 必须 ≥ 1；为 0 说明 6.1 没生效（回第 6.1 节并重新清缓存） |
| 2 | `tail -3 $(ls -t ~/SSR/logs/smoke1-*.log \| head -1)` | 取最新那份单题日志的末尾，应看到 `Graph QA Accuracy: 1.0000 (1/1)` 之类的汇总，而不是 traceback |

### 10.4 二十题（几十秒）

```bash
bash ~/SSR/tools/run_ssr.sh smoke20
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `bash ~/SSR/tools/run_ssr.sh smoke20` | 跑 20 题（`--sample-size 20 --concurrency 8`），固定随机种子，每次同一批；缓存由启动器自动清 |

**判据**（三项都满足才继续）：

| 判据 | 期望 |
|---|---|
| 不崩 | 日志末尾没有 `Traceback`，没有 `does not match format` |
| 出总量 | 有 `Graph QA Accuracy: 0.xxxx (n/20)` |
| 有标签 | `grep -c '<fact>' artifacts/llm_cache.jsonl` 明显大于 0 |

**耗时参考**（未单独计时，别拿它当指标）：QA 部分并发 8 约 0.37 s/题，20 题约 8 s；同一批 20 题在并发 16 下实测总耗时 50 s（2.54 s/题，见 10.6 的退化说明）。所以正常是几十秒量级，若超过 2 分钟说明并发或缓存有问题。

**这一节只看"通不通"，不看"准不准"**：20 题的 95% 置信区间约 ±20 个百分点（0.4 的真实区间是 0.22–0.61），和论文数字没有可比性。要基线请走 10.5 的 200 题。

### 10.5 两百题（约 5 分钟，出基线数字）

```bash
bash ~/SSR/tools/run_ssr.sh smoke200
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `bash ~/SSR/tools/run_ssr.sh smoke200` | 跑 200 题（`--sample-size 200 --concurrency 8 --random-seed 7`）；并发固定 8、种子固定 7，便于两次改动前后对比；缓存由启动器自动清 |

**判据**：

| 判据 | 期望 |
|---|---|
| 不崩 | 无 `Traceback`、无 `invalid literal for int()` |
| 出总量 | `Graph QA Accuracy: 0.xxxx (n/200)`，`n` 在 100 以上 |
| 单题耗时 | Step 3 的 `s/it` ≤ 1.0（慢于 1.0 就是并发退化，见 10.6） |

**实测基线（2026-11-01，并发 8，seed 7）**：200 题共 261 s（Step 2 抽取 102 s、Step 3 QA 159 s），1.31 s/题，`Graph QA Accuracy: 0.6950 (139/200)`，`atype entity: 0.71875`、`atype time: 0.60000`。换种子数字会明显浮动（同一个流程 seed 42 的 20 题只有 0.40），所以**别用单次结果和论文数字硬比**，全量跑完的总体数才是可报的。


### 10.6 全量（54584 题，约 20 小时）

用 tmux 跑，断开 ssh 也不会停：

```bash
tmux new -d -s full -c ~/SSR 'bash ~/SSR/tools/run_ssr.sh full'
tmux attach -t full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s full -c ~/SSR 'bash ~/SSR/tools/run_ssr.sh full'` | 在名为 `full` 的后台会话里跑全量（省略 `--sample-size` 就是全量）；`-c ~/SSR` 指定工作目录；旧缓存由启动器自动清 |
| 2 | `tmux attach -t full` | 进去看实时进度条；按 `Ctrl+B` 再按 `D` 脱离，**会话和实验继续跑**，之后再用这条命令回来 |

**为什么用 tmux 而不是 nohup**：tmux 里的进程本身就是脱离终端的，命令行就是一条普通的 `bash ~/SSR/tools/run_ssr.sh full`，不存在"后台找不到 `python`"这种事；进度条、报错、"跑完最后一行"都留在 pane 里可以往回翻（`Ctrl+B` `[`）。

**启动后 20 秒内先做这次体检**（别等一晚上才发现它早就退了）：

```bash
tmux ls
pgrep -af "run_ssr.sh full" | head -3
tail -c 600 $(ls -t ~/SSR/logs/full-*.log | head -1) | tr '\r' '\n' | tail -4
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux ls` | 应看到 `full: 1 windows (created …)` |
| 2 | `pgrep -af "run_ssr.sh full"` | 有输出 = 启动器活着；无输出 = 已退出，进 pane 看报错 |
| 3 | `tail -c 600 $(ls -t ~/SSR/logs/full-*.log \| head -1) \| tr '\r' '\n' \| tail -4` | 取最新全量日志的尾部；正常是 `Load cached text retriever` 或 `Run Text Retrieval` / `Extract graph patterns` 进度条 |

日志文件名形如 `logs/full-1101-1510.log`（模式-月日-时分），跑完后也可以用 `less` 逐页看，不必依赖 tmux 回滚。

**Step 1 的耗时判据**：`Searching similar facts` 一共 427 批（每批 128 题），时间上限约 **每批 6–8 秒**（Step 1 全部约 40 分钟）。如果显示 `90 s/it`、`预计 10 小时`，说明 OpenMP 线程没限住（节点 128 核、load 100+，线程互抢比单线程还慢）：确认启动器里有 `export OMP_NUM_THREADS=8`，或手工跑时自己先 export。实测同一批 128 题：1 线程 6.2 s、8 线程 6.5 s、32 线程 10.6 s、128 线程 **88.9 s**。

**跑起来之后应该长这样**（2026-11-01 15:52 实测，日志 `logs/full-1101-1538.log`）：`tmux ls` 里有 `full`；Step 1 稳定在 `6.5 s/it`；`nvidia-smi` 显存到 **45996 MiB**；`ls ~/SSR/artifacts` 此时**还没有** `llm_cache.jsonl`——Step 1 只做编码和检索、不调用 LLM，这个文件要到 Step 2 才开始写，**不要据此判断"没生效"**。

**不要把并发往上调**。实测同 20 条 QA：并发 8 是 7.3 s（0.37 s/题），并发 16 变成 74.5 s（3.73 s/题）——有的请求会一直自言自语到 4096 token 上限才停（vLLM `/metrics` 里 `finished_reason="length"` 飙升），还会夹带乱码。并发越大不是越快，而是排队长 + 单条跑满。

**耗时**：实测 1.31 s/题（Step 2 抽取约 0.5 s/题 + Step 3 QA 约 0.8 s/题），5.4584 万题约 **20 小时**；首次还会多花约 10 分钟建文本索引。

**断点续跑**：中断后不要清缓存，用 `SSR_KEEP_CACHE=1 tmux new -d -s full -c ~/SSR 'bash ~/SSR/tools/run_ssr.sh full'` 重新启动，`artifacts/llm_cache.jsonl` 会复用已完成的 LLM 调用。判断是否真的进了续跑模式：日志里应显示 `keep llm_cache.jsonl (resume mode: SSR_KEEP_CACHE=1)`，而不是 `cleared`。一旦改过 prompt，旧缓存键全部失配，等于从零开始，这时不要用这个前缀。

### 本节验收

- 单题阶段 `grep -c '<fact>'` ≥ 1；
- 20 题阶段三项判据全过；
- 200 题阶段 `Graph QA Accuracy` 的样本数 `n ≥ 100`；
- 全量日志末尾出现 `Hit@1 … Hit@100` 与 `overall accuracy`。

---

## 11. 常见错误对照表

| 报错 / 现象 | 真因 | 处置 |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | 用了 `python -m` 模块方式启动，而模块名在 `~/SSR` 下并不存在 | 改用启动器 `bash ~/SSR/tools/run_ssr.sh smoke1`（10.1），它固定用 `python src/run_pipeline.py` 的脚本方式 |
| `python: command not found` | 新 shell 没激活 conda，`python` 不在 PATH | 改用启动器（10.1）；或 `conda activate ssr`；或绝对路径 `~/miniconda3/envs/ssr/bin/python` |
| `nohup: failed to run command 'python': No such file or directory` | 后台 `nohup` 不继承交互 shell 的环境，找不到 `python` | 不要用 nohup 起实验，统一走 10.6 的 tmux + `run_ssr.sh`（启动器内部是绝对路径） |
| `ValueError: time data 'start_date' does not match format '%Y-%m-%d %H:%M:%S'` | 没做 6.3 的容错，模型输出了占位词 | 回第 6.3 节打补丁 |
| `ValueError: invalid literal for int() with base 10: '?2011'` | 打了 6.3 但 `try:` 只包住 `datetime(...)`，`int(y)` 先抛了 | 按 6.3 最新版把 `try:` 提到 `y, m, d = raw_time.split("-")` 之前 |
| `IndentationError: unindent does not match any outer indentation level` | 粘贴代码时缩进多了/少了空格（常见于 `@staticmethod` 行） | `python -m py_compile` 定位行号，补成 4 空格 |
| `error: unrecognized arguments: False` | 三个 `--run-*` 是 `action="store_true"` 开关，后面跟 `False` 会被当成位置参数 | 只写开关名，不带值 |
| `Connection error` / `401 Unauthorized` 指向 `api.openai.com` | 漏了 `OPENAI_BASE_URL`（`config.py` 默认值就是官方地址） | 三个变量一起 export |
| `Connection refused` 连 `127.0.0.1:8000` | vLLM 没起，或起在另一台机器 | `pgrep -af "vllm serve"`、`ss -ltnp \| grep 8000`，在 3-168 上重起 |
| Step 2 进度条显示几千 it/s | 全部命中旧 `llm_cache.jsonl`，不是加速 | 启动器每次已自动清；仍出现说明这次带了 `SSR_KEEP_CACHE=1`，或绕开启动器手工跑了 `python`（回第 10.1 节） |
| pattern 恒为 `("?","?","?")`、准确率 0.2 左右 | 6.1 的三行没生效（`gp_prompt.txt` 末尾没有 `<fact>` 那两行） | 回 6.1 重打（幂等），再清缓存重跑 |
| `FileNotFoundError: ./datasets/MultiTQ/...` | 不在 `~/SSR` 下执行 | `cd ~/SSR` 重跑 |
| `FileNotFoundError: .../kg/entity2id.json` | 数据解压层级不对，id 映射在外层 | 按 4.3 的 `mv` 挪进 `kg/` |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | `vllm` 环境里 transformers 版本太新 | 装 `transformers==4.51.3` |
| 日志里出现 `Graph QA Accuracy: 0.0xxx` 且题量正常 | 严格匹配导致：`is_correct` 用 `==` 比对，下划线/空格/日期格式差异都判错 | 看 `llm_cache.jsonl` 里 QA 的原始输出，人工比对格式差异 |
| 回答里出现 `ERROR` | `query_process.py` 在响应不含 `So the answer is` 时直接返回 `ERROR` | 做 6.2 的三行（`qa_prompt.txt`），并确认 `max_tokens` 没被截断 |
| 单题耗时突然变成 2.5–4 s、输出夹乱码、某条一直不停 | 并发开太大（16 及以上）导致请求排队、个别请求跑满 4096 token | 把 `--concurrency` 降到 8 重跑 |
| Step 3 的进度条 `s/it` 从 0.4 跳到 2 以上 | 同上，属于并发退化，不是模型变慢 | `--concurrency 8`；用 `/metrics` 看 `finished_reason="length"` 是否飙升 |
| Step 1 的 `Searching similar facts` 显示 `90+ s/it`、预计 10 小时以上 | OpenMP 默认按 128 核开线程，而节点是共享的（load 100+），线程互相抢 CPU，平扫反而慢约 14 倍 | 启动器已 `export OMP_NUM_THREADS=8`（实测每批 128 题从 89 s 降到 6.2 s，Step 1 由约 11 小时变约 40 分钟）；手工跑时记得同样设置 |

---

## 12. tmux 速查与换机器

### 12.1 tmux 速查

| 目的 | 操作 |
|---|---|
| 看有哪些会话 | `tmux ls` |
| 进入会话 | `tmux attach -t <名字>` |
| 脱离而不杀进程 | `Ctrl+B` 松开，再按 `D` |
| 结束会话 | `tmux kill-session -t <名字>` |
| 新建后台会话跑命令 | `tmux new -d -s <名字> -c ~/SSR '<命令>'` |
| 在会话内滚动看历史 | `Ctrl+B` 然后 `[`，`q` 退出 |
| 看后台服务日志 | `tail -f ~/SSR/logs/vllm-3-168.log` |
| 看是不是"已守护"的进程 | `ps -o pid,ppid,tty,cmd -p <pid>`（`PPID=1` + `TTY=?` 就是脱离了终端） |
| 查端口占用 | `ss -ltnp \| grep 8000` |

**要不要用 tmux 的判定**：

| 任务 | 需要 tmux 吗 | 原因 |
|---|---|---|
| vLLM 服务 | 不需要 | 已经用 `nohup` + `disown` 守护（`PPID=1`、`TTY=?`），关终端不受影响 |
| `run_pipeline.py` 全量 | **需要**（10.6） | 前台进程，断 ssh 就死；且 tqdm 进度条接管道会退化成乱刷。用 `tmux new -d -s full -c ~/SSR 'bash ~/SSR/tools/run_ssr.sh full'` |
| 几秒的检查命令 | 不需要 | 当场出结果 |

tmux 里跑实验时不需要 `export` 任何变量、也不需要 `(ssr)` 前缀——`run_ssr.sh` 自己设好了（见 10.1）。

### 12.2 换机器（共享 NFS）

```bash
df -h ~/SSR
ls ~/SSR && conda env list
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `df -h ~/SSR` | 确认新机也挂同一个 NFS（代码/数据/模型/conda 都在盘上，无需重装） |
| 2 | `ls ~/SSR && conda env list` | 确认目录与两个环境都在 |

在旧机停服务，避免两台机同时抢同一张卡/同一个编译缓存：

```bash
pkill -f "vllm serve"
sleep 5 && pgrep -af "vllm serve" || echo "vllm stopped"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `pkill -f "vllm serve"` | 停掉旧机上的 vLLM |
| 2 | `pgrep -af "vllm serve" \|\| echo ...` | 确认已经停干净（无输出即停好） |

然后在新机执行第 9 节启动（日志名换成 `logs/vllm-<机器标识>.log`），再执行第 10 节。

### 12.3 两条共享盘禁忌

- `artifacts/` 与 `llm_cache.jsonl` 在共享盘上，**两台机不要同时跑同一个实验**；
- 旧机的 vLLM 未停就起新机的，会导致两边都抢 48 GB 卡（`CUDA out of memory` 或显存直接 0）。

### 本节验收

- `tmux ls` 能看到你的会话，`tmux attach` 能进能脱；
- 换机后 `df -h ~/SSR` 仍指向 `172.30.20.24:/ltaidc`；
- 旧机 `pgrep -af "vllm serve"` 无输出。
