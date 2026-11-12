# SSR：MultiTQ 时序知识图谱问答复现

基于论文 **SSR: Structured Subgraph Retrieval for Temporal Knowledge Graph Question Answering with LLMs** 的公开实现，在 MultiTQ 上使用 **BGE-M3 + Qwen2.5-Instruct-AWQ + vLLM** 完成文本检索、结构化子图检索和问答评测。

本仓库是面向本地千问推理的复现与部署版本。算法来源见[作者仓库](https://github.com/zhangli-coding/SSR)，本仓库地址为 [2200170080/SSR](https://github.com/2200170080/SSR)。主要运行环境为 Linux 与单张 RTX A6000 48 GB；运行过程使用预训练权重执行推理和评测。

![SSR 方法框架](figure/main.png)

**已有环境：**直接阅读[快速开始](#1-快速开始)。**首次部署：**按第 3–5 节准备环境、数据、模型和索引，再回到第 1 节。常用操作见[停止服务与释放显存](#73-停止实验与-vllm-释放显存)、[缓存与续跑](#9-缓存续跑与完成判据)、[目录维护](#10-历史归档与日常维护)、[GitHub 同步](#122-github-自动同步完整流程)和[故障排查](#11-常见问题)。

## 0. 项目概览与实验结果

### 方法流程

| 阶段 | 输入与处理 | 输出 |
|---|---|---|
| 文本检索 | BGE-M3 编码问题与 KG 事实，通过 FAISS 检索相近事实 | 每题前 100 条文本候选 |
| 图模式解析 GP | LLM 结合问题与前 50 条文本候选，生成实体、关系和时间约束 | `<fact>` 与 `<time>` 标签 |
| 图检索 | 按图模式检索 KG，并处理实体／关系对齐与日期过滤 | 符合模式的事实候选 |
| 问答 QA | 从图候选中选择证据；图候选为空时使用文本检索候选 | 实体或年／月／日答案 |
| 评测 | 提取最终答案，与标准答案逐题比较 | Overall、Entity、Time 准确率 |

BGE 使用首 token 表示、L2 归一化和 FAISS `IndexFlatIP`；归一化向量的内积对应余弦相似度。实现位置见 [TextRetriever](src/retriever/text_retriever.py)。

### 已完成的全量结果

| 配置 | 正确数 / 总数 | Graph QA Accuracy | Entity | Time |
|---|---:|---:|---:|---:|
| Qwen2.5-14B-Instruct-AWQ：GP + QA | 40066 / 54584 | 0.7340 | 0.67990 | 0.86590 |
| Qwen2.5-32B-Instruct-AWQ：GP + QA | 45321 / 54584 | **0.8303** | **0.78044** | **0.95178** |

14B 的依据是[历史全量日志](https://github.com/2200170080/SSR/blob/a8d634f514baf1d7c00506f90c297b45624b6ea6/artifacts/qwen2.5-14b-awq/history/full-20261103/full.log)及对应终端输出；32B 的依据是[历史摘要](https://github.com/2200170080/SSR/blob/a8d634f514baf1d7c00506f90c297b45624b6ea6/artifacts/qwen2.5-32b-awq/history/full-20261104/summary.json)、完成记录和对应终端输出。两次历史全量均保留 54584 条原始问题记录，包含重复问句。32B 比该 14B 历史结果多答对 5255 条，总分高约 9.63 个百分点；这是两次运行结果之差，不能据此推算 GPT 的预期成绩。

32B 历史摘要仍记录 **18 条缺少最终答案标记的回答**。以上是实际评测结果，不表示所有输出格式问题已经消失；也不是目录迁移后新启动器已经重新跑完的成绩。

### 全流程与跳过条件

| 场景 | 阅读顺序 | 操作原则 |
|---|---|---|
| 已部署环境，准备新实验 | 1 → 7 → 9 | 先预检，再选择一个模型启动 |
| 全新服务器 | 3 → 4 → 5 → 1 | 先建索引，再启动 Qwen |
| 中断后的同配置续跑 | 9 | 保留原运行目录与请求缓存 |
| 查看已完成成绩 | 0、9、10 | 读取摘要和日志即可 |
| 迁移到另一台服务器 | 2、3、10、12 | 核对未上传资产与本机模型服务 |

## 1. 快速开始

以下命令在**服务器终端**执行，默认项目位于 `$HOME/SSR`，两套 Conda 环境、模型及四份检索资产已准备好。单张 A6000 同一时刻运行一份全量任务。

### 1.1 只做启动前检查

只部署一种模型时，执行对应的检查行；两种模型都已部署时，可依次检查两套入口。

```bash
cd "$HOME/SSR"
bash scripts/full-evaluation/run-14b.sh --check
bash scripts/full-evaluation/run-32b.sh --check
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入项目根目录 |
| 2 | `bash scripts/full-evaluation/run-14b.sh --check` | 检查 14B 的环境、输入、权重及当前服务归属 |
| 3 | `bash scripts/full-evaluation/run-32b.sh --check` | 检查 32B；不启停服务、不创建实验目录 |

检查结果 `target` 表示端口上的模型与本次所选模型一致；`other_ssr` 表示在线的是本项目另一种模型；`stopped` 表示没有已就绪服务；`blocked` 表示需要先排查服务或进程归属。`other_ssr` 和 `stopped` 都可通过静态预检，正式启动时再处理模型服务。

### 1.2 启动 32B 全量评测

```bash
cd "$HOME/SSR"
bash scripts/full-evaluation/run-32b.sh
tmux attach -t ssr-32b-full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入运行根目录 |
| 2 | `bash scripts/full-evaluation/run-32b.sh` | 在 tmux 中检查、准备 32B 服务，并执行完整 54584 题 |
| 3 | `tmux attach -t ssr-32b-full` | 进入任务会话，查看预检、服务启动及运行进度 |

### 1.3 选择 14B 时使用另一套入口

```bash
cd "$HOME/SSR"
bash scripts/full-evaluation/run-14b.sh
tmux attach -t ssr-14b-full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入运行根目录 |
| 2 | `bash scripts/full-evaluation/run-14b.sh` | 使用 14B 完成 GP 与 QA，独立保存结果和缓存 |
| 3 | `tmux attach -t ssr-14b-full` | 查看 14B 任务 |

两套全量入口选择一套运行。按 **Ctrl+b**，松开后按 **d**，可以脱离 tmux 并保持后台任务；会话内按 **Ctrl+C** 会中断任务。

启动命令返回“已创建启动会话”只说明 tmux 已创建。是否通过预检、模型是否能生成、全量是否完成，应继续查看会话和运行结果。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| 对应入口加 `--check` | 输出静态预检通过，服务状态不是 `blocked` |
| `tmux attach -t ssr-32b-full` 或 `ssr-14b-full` | 可看到本次启动过程或已退出窗格的结果 |
| 会话中的“模型实际生成检查通过” | 对应模型服务已通过一次真实生成检查 |

## 2. 目录结构与运行约定

```text
SSR/
├── README.md
├── SSR_GitHub自动同步.md
├── LICENSE
├── datasets/MultiTQ/
│   ├── kg/                         # KG、entity2id.json、relation2id.json
│   ├── questions/test.json         # 54584 条原始问题
│   └── prompt/
│       ├── Qwen/                   # gp_prompt.txt、qa_prompt.txt
│       ├── Llama/                  # gp_prompt.txt、qa_prompt.txt
│       └── GPT/                    # 作者原版 gp_prompt.txt、qa_prompt.txt
├── models/                         # 本机权重，Git 忽略
│   ├── bge-m3/
│   ├── qwen14b-awq/
│   └── qwen32b-awq/
├── src/
│   ├── run_pipeline.py             # 原生检索与评测入口
│   ├── config.py                   # 数据、模型、接口默认配置
│   ├── retriever/                  # 文本检索与图检索
│   └── utils/                      # 请求、提示词、采样、答案解析、评测
├── scripts/
│   ├── full-evaluation/
│   │   ├── run-14b.sh
│   │   ├── run-32b.sh
│   │   └── launcher.py             # 预检、模型服务、tmux、缓存续跑
│   └── github-sync/
│       ├── autosync.sh
│       └── autosync.log            # 同步器日志，Git 忽略
├── artifacts/
│   ├── shared/
│   │   ├── indexes/
│   │   │   ├── multitq_text_index.pkl
│   │   │   ├── multitq_graph_retriever.pkl
│   │   │   ├── MultiTQ_entity_embeddings.npy
│   │   │   └── MultiTQ_relation_embeddings.npy
│   │   ├── generation-config/generation_config.json
│   │   ├── locks/                  # 运行锁，Git 忽略
│   │   └── history/                # 旧模型对照等归档
│   ├── qwen2.5-14b-awq/
│   │   ├── history/full-20261103/
│   │   ├── runs/                   # 该模型的新实验
│   │   └── service/                # 该模型的 vLLM 日志
│   └── qwen2.5-32b-awq/
│       ├── history/full-20261104/
│       ├── runs/
│       └── service/
└── figure/main.png
```

树中也列出了运行时生成或被 Git 忽略的目录，新克隆中不一定存在。两份 `.npy` 是 BGE-M3 的实体／关系向量，已与两个 `.pkl` 统一放在 `artifacts/shared/indexes/`。图检索和启动器均按此位置读取，不再从项目根目录读取这两份向量。

| 约定 | 路径或值 |
|---|---|
| 项目根目录 | `$HOME/SSR`；本文命令均先进入此处 |
| 实验解释器 | `$HOME/miniconda3/envs/ssr/bin/python` |
| 模型服务可执行文件 | `$HOME/miniconda3/envs/vllm/bin/vllm` |
| 提示词目录 | `datasets/MultiTQ/prompt/{Qwen,Llama,GPT}/`，每套各含 GP 与 QA |
| 本地接口 | `http://127.0.0.1:8000/v1`，客户端 key 为 `dummy` |
| 14B / 32B 模型别名 | `qwen14b` / `qwen32b-gp` |

普通相对路径以当前工作目录为基准；`config.py` 中的 `PROJECT_ROOT` 不会自动把所有默认路径转换成绝对路径。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `ls datasets/MultiTQ/prompt/{Qwen,Llama,GPT}` | 每套含 `gp_prompt.txt` 和 `qa_prompt.txt` |
| `ls artifacts/shared/indexes` | 已部署环境有两个 `.pkl` 和两份 `.npy` |
| `ls scripts/full-evaluation` | 有两份 shell 入口及共享 `launcher.py` |

## 3. 首次部署环境

已有环境且预检通过时跳过本节安装。以下版本沿用已有部署记录，不是本次重新安装的现场结果，也不是完整依赖锁文件。主机需预先具备 NVIDIA 驱动以及 `git`、`curl`、`tmux`、`ss`、`nvidia-smi`。

| 环境 | 用途 | 已用版本组合 |
|---|---|---|
| `ssr` | BGE 编码、FAISS、图检索、评测 | Python 3.10；torch 2.4.1 / cu121；NumPy 1.26.4；faiss-cpu 1.8.0.post1；transformers 4.44.2 |
| `vllm` | Qwen 的 OpenAI 兼容服务 | Python 3.10；vLLM 0.8.5；transformers 4.51.3；torch 使用 vLLM 自身匹配依赖，历史为 2.6.0 |

### 3.1 首次安装 Miniconda

仅当 `$HOME/miniconda3/bin/conda` 不存在时执行。安装方法参考 [Miniconda 官方文档](https://www.anaconda.com/docs/getting-started/installation)。

```bash
cd "$HOME"
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME"` | 进入用户目录 |
| 2 | `curl -fL ... -o Miniconda3-latest-Linux-x86_64.sh` | 下载 Linux x86_64 安装器 |
| 3 | `bash ... -b -p "$HOME/miniconda3"` | 安装到约定位置 |
| 4 | `source .../conda.sh` | 加载当前 shell 的 Conda 函数 |
| 5 | `conda --version` | 验证安装入口 |

### 3.2 创建两套环境

下面分别创建实验和服务环境。已有同名环境时先检查，不重复创建或覆盖。PyTorch wheel 版本依据[官方历史版本页面](https://pytorch.org/get-started/previous-versions/)，vLLM 安装约束参考 [v0.8.5 文档](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)。

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n ssr python=3.10 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
"$HOME/miniconda3/envs/ssr/bin/python" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
"$HOME/miniconda3/envs/ssr/bin/python" -m pip install numpy==1.26.4 faiss-cpu==1.8.0.post1 transformers==4.44.2 "tokenizers<0.20" sentencepiece protobuf openai==1.109.1 tqdm huggingface_hub==0.36.2
conda create -n vllm python=3.10 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
"$HOME/miniconda3/envs/vllm/bin/python" -m pip install vllm==0.8.5
"$HOME/miniconda3/envs/vllm/bin/python" -m pip install transformers==4.51.3 "huggingface_hub<1.0"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `source .../conda.sh` | 加载 Conda |
| 2 | `conda create -n ssr ...` | 创建 Python 3.10 实验环境 |
| 3 | `.../ssr/bin/python -m pip install torch==2.4.1 ...` | 安装实验侧 CUDA 12.1 wheel |
| 4 | `.../ssr/bin/python -m pip install numpy==1.26.4 ...` | 安装检索、接口和下载依赖 |
| 5 | `conda create -n vllm ...` | 创建独立服务环境 |
| 6 | `.../vllm/bin/python -m pip install vllm==0.8.5` | 安装服务及其匹配依赖 |
| 7 | `.../vllm/bin/python -m pip install transformers==4.51.3 ...` | 使用已有部署的分词器兼容版本 |

### 3.3 验证环境与首次克隆

克隆命令仅用于尚无 `$HOME/SSR` 仓库的新环境；已有仓库直接进入并检查。不要在实验运行期间强制检出或还原源码。

```bash
"$HOME/miniconda3/envs/ssr/bin/python" -c "import torch,faiss,numpy,transformers,openai; print(torch.__version__,faiss.__version__,numpy.__version__,transformers.__version__,openai.__version__); print('CUDA:',torch.cuda.is_available())"
"$HOME/miniconda3/envs/vllm/bin/python" -c "import vllm,torch,transformers; print(vllm.__version__,torch.__version__,transformers.__version__); print('CUDA:',torch.cuda.is_available())"
git clone https://github.com/2200170080/SSR.git "$HOME/SSR"
cd "$HOME/SSR"
git log -1 --oneline
git status --short
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `.../ssr/bin/python -c ...` | 验证实验依赖与 CUDA |
| 2 | `.../vllm/bin/python -c ...` | 验证服务依赖与 CUDA |
| 3 | `git clone ... "$HOME/SSR"` | 首次获取本仓库代码、数据和受跟踪向量 |
| 4 | `cd "$HOME/SSR"` | 进入运行根目录 |
| 5 | `git log -1 --oneline` | 记录实际使用的代码版本 |
| 6 | `git status --short` | 查看工作区差异 |

### 本节验收

| 检查命令 | 判据 |
|---|---|
| 两套解释器的导入检查 | 依赖可导入，CUDA 为 True |
| `tmux -V`、`ss --version`、`nvidia-smi` | 所需系统工具可用 |
| `ls "$HOME/SSR/src/run_pipeline.py"` | 找到原生入口 |

## 4. 数据与模型权重

### 4.1 MultiTQ 输入

本仓库已跟踪 MultiTQ 的 KG、ID 映射、测试集与三套提示词。数据来源为 [MultiTQ 作者项目](https://github.com/czy1999/MultiTQ)，也可参考[原 README 提供的 Hugging Face 数据链接](https://huggingface.co/datasets/chenziyang/MultiTQ)。先检查现有文件，避免重复下载或覆盖生效提示词。

```bash
cd "$HOME/SSR"
"$HOME/miniconda3/envs/ssr/bin/python" - <<'PY'
import json
from pathlib import Path
p = Path('datasets/MultiTQ')
for name in ('kg/full.txt', 'kg/entity2id.json', 'kg/relation2id.json',
             'questions/test.json'):
    assert (p / name).is_file(), f'MISSING: {p / name}'
for family in ('Qwen', 'Llama', 'GPT'):
    for stage in ('gp', 'qa'):
        path = p / 'prompt' / family / f'{stage}_prompt.txt'
        assert path.is_file(), f'MISSING: {path}'
with (p / 'kg/full.txt').open(encoding='utf-8') as stream:
    facts = sum(1 for line in stream if line.strip())
questions = json.loads((p / 'questions/test.json').read_text(encoding='utf-8'))
print('KG facts:', facts, 'test questions:', len(questions))
assert facts == 461329 and len(questions) == 54584
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定数据相对路径 |
| 2 | `.../ssr/bin/python - <<'PY' ...` | 检查六个输入文件并核对部署基线规模 |

原项目还介绍了 [TimelineKGQA](https://github.com/PascalSun/TimelineKGQA/tree/main/Datasets)，其中使用 CronQuestions KG 部分。本仓库当前打通和报告的是 **MultiTQ**，没有在这里提供 TimelineKGQA 的已完成结果或专用启动器。

### 4.2 下载 BGE-M3

模型来源：[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)。已有完整模型时跳过下载。镜像地址仅影响下载，预检和实验从本地路径读取。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HOME/SSR/.hf_cache"
"$HOME/miniconda3/envs/ssr/bin/huggingface-cli" download BAAI/bge-m3 --local-dir "$HOME/SSR/models/bge-m3" --exclude "imgs/*" "onnx/*"
"$HOME/miniconda3/envs/ssr/bin/python" -c "from transformers import AutoTokenizer; print(type(AutoTokenizer.from_pretrained('$HOME/SSR/models/bge-m3',local_files_only=True)).__name__)"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `export HF_ENDPOINT=...` | 选择下载镜像 |
| 2 | `export HF_HOME=...` | 将下载缓存放在项目内 |
| 3 | `.../huggingface-cli download BAAI/bge-m3 ...` | 下载编码模型与本地分词器文件 |
| 4 | `.../ssr/bin/python -c ...` | 验证分词器能离线加载 |

### 4.3 下载所选 Qwen 模型

模型来源：[14B AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)、[32B AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)。以下沿用服务器曾使用的 ModelScope 下载方式；已有工具和权重时跳过。只运行 32B 无需额外下载 14B，反之亦然。

```bash
"$HOME/miniconda3/envs/ssr/bin/python" -m pip install modelscope
"$HOME/miniconda3/envs/ssr/bin/modelscope" download --model Qwen/Qwen2.5-14B-Instruct-AWQ --local_dir "$HOME/SSR/models/qwen14b-awq"
"$HOME/miniconda3/envs/ssr/bin/modelscope" download --model Qwen/Qwen2.5-32B-Instruct-AWQ --local_dir "$HOME/SSR/models/qwen32b-awq"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `.../ssr/bin/python -m pip install modelscope` | 仅在缺少下载工具时安装 |
| 2 | `.../modelscope download --model ...14B...` | 选择 14B 时下载到约定目录 |
| 3 | `.../modelscope download --model ...32B...` | 选择 32B 时下载到约定目录 |

已有部署记录中，14B 为 3 个 safetensors 分片，32B 为 5 个。以各自 `model.safetensors.index.json` 引用的文件为准。启动器检查配置、分词器、AWQ 类型、分片张量头与偏移；这属于结构检查，不等同于重新计算全部权重的内容哈希。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| 第 4.1 节计数检查 | KG 461329 条非空事实，测试集 54584 条 |
| BGE 分词器离线加载 | 返回本地分词器类，无需在线补文件 |
| 模型索引及对应 `--check` | 所选模型配置与全部分片可读 |

## 5. 首次预建共享索引

**仅用于首次部署或明确缺失的资产。** 已有四份正确资产时直接跳过，不因目录整理而重建。完整 KG 的 BGE 编码会占用较多显存，预建期间应保持本项目 Qwen 服务停止、相关管线退出，并确认 GPU 资源可用。

### 5.1 查看资产与显存

```bash
cd "$HOME/SSR"
ls -lh artifacts/shared/indexes
nvidia-smi
ss -ltnp 'sport = :8000'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入运行根目录 |
| 2 | `ls -lh artifacts/shared/indexes` | 确认两个 `.pkl` 和两份 `.npy` 是否已齐全 |
| 3 | `nvidia-smi` | 核对实际 GPU 占用 |
| 4 | `ss -ltnp 'sport = :8000'` | 查看是否有在线模型服务 |

若已有本项目服务，按[第 7.3 节](#73-停止实验与-vllm-释放显存)先停止实验和 vLLM，确认资源释放后再建库。无法确认归属的进程应先核对，不根据旧 PID 或进程名批量结束其他任务。

### 5.2 首次生成文本索引

只启用文本检索阶段；`--sample-size 1` 只限制测试问题数量，KG 仍完整编码建库，此过程不请求 LLM。已有文本索引时会加载它；本命令不强制覆盖。

```bash
cd "$HOME/SSR"
mkdir -p artifacts/shared/indexes
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$HOME/miniconda3/envs/ssr/bin/python" -u src/run_pipeline.py \
  --run-text-retrieval \
  --text-retriever-model-name "$PWD/models/bge-m3" \
  --text-index-path artifacts/shared/indexes/multitq_text_index.pkl \
  --device cuda:0 --sample-size 1 --random-seed 42 --text-top-k 100
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定输入与缓存路径 |
| 2 | `mkdir -p artifacts/shared/indexes` | 建立共享索引目录 |
| 3 | `export OMP_NUM_THREADS=8 ...` | 限制检索侧 CPU 线程 |
| 4 | `.../ssr/bin/python -u src/run_pipeline.py --run-text-retrieval ...` | 完整预建文本索引，再验证一次问题检索 |

### 5.3 首次生成图缓存与缺失向量

复用现有 `build_graph_retriever()`，不启动 GP 或 QA。已跟踪的 `.npy` 齐全时直接加载，缺失时由原编码逻辑生成到同一共享目录。

```bash
cd "$HOME/SSR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$HOME/miniconda3/envs/ssr/bin/python" - <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, 'src')
from config import DatasetConfig
from run_pipeline import build_graph_retriever
d = DatasetConfig()
args = SimpleNamespace(
    graph_cache_path=d.graph_cache_path,
    entity_path=d.entity_path,
    relation_path=d.relation_path,
    kg_path=d.kg_path,
    text_retriever_model_name=str(Path.cwd() / 'models/bge-m3'),
    device='cuda:0', dataset_name=d.name, force_reload=False,
)
build_graph_retriever(args)
print('Graph cache ready')
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定源码导入和运行路径 |
| 2 | `export OMP_NUM_THREADS=8 ...` | 保持建库线程设置 |
| 3 | `.../ssr/bin/python - <<'PY' ...` | 创建或加载图缓存，并准备实体／关系向量 |

两个预建进程退出后再启动 Qwen。加载共享索引仍需 BGE 编码问题、完成实体／关系对齐，预建的作用是避开完整 KG 编码峰值。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `ls -lh artifacts/shared/indexes` | 四份检索资产均存在且非空 |
| 第 5.2、5.3 节输出 | 分别出现 `Done` 和 `Graph cache ready` |
| 对应模型 `--check` | NPY 行数、维度和 float32 类型符合当前 ID 映射与 BGE 配置 |

## 6. 提示词与复现范围

启动器按模型选择提示词目录。预检和实际运行读取同一对文件，运行指纹记录两份文件的路径和 SHA-256。

| 提示词目录 | 文件 | 用途 |
|---|---|---|
| Qwen | [GP](datasets/MultiTQ/prompt/Qwen/gp_prompt.txt) / [QA](datasets/MultiTQ/prompt/Qwen/qa_prompt.txt) | Qwen 14B、32B 启动器和原生入口默认值 |
| Llama | [GP](datasets/MultiTQ/prompt/Llama/gp_prompt.txt) / [QA](datasets/MultiTQ/prompt/Llama/qa_prompt.txt) | Llama 3.1 8B 启动器 |
| GPT | [GP](datasets/MultiTQ/prompt/GPT/gp_prompt.txt) / [QA](datasets/MultiTQ/prompt/GPT/qa_prompt.txt) | 作者原版模板，供 GPT 实验显式指定 |

Qwen 保留原本地提示词。Llama 目录在迁移时使用同一对文件初始化，之后可独立维护。新实验使用所选目录下的文件。提示词迁移前的运行指纹仍记录原路径，需为新配置创建独立运行目录。

GPT 文件逐字节保留作者仓库固定提交 `1003f62206869613f0e22084bbf2b5620ad65cff` 中的原版：

- [GP 原文件](https://github.com/zhangli-coding/SSR/blob/1003f62206869613f0e22084bbf2b5620ad65cff/src/datasets/MultiTQ/prompt/gp_prompt.txt)，SHA-256 `8bf56862b15df9d3efbfd1e0a4932244f13a019645a5bf5918a1e1f190b34d75`
- [QA 原文件](https://github.com/zhangli-coding/SSR/blob/1003f62206869613f0e22084bbf2b5620ad65cff/src/datasets/MultiTQ/prompt/qa_prompt.txt)，SHA-256 `cdbc2b264b7299fcf2d44c1b1a54336f91abfdbea3f16421ddc0ad3a22ba9be0`

| 项目 | 当前实现 |
|---|---|
| GP 协议 | `<fact>头实体; 关系; 尾实体</fact>`、`<time>起始日期; 结束日期</time>`；未知槽位使用 `?` |
| Slot selection | 保留问题中已指定的实体和关系；只对待求或未指定槽位用问号；first/last 问题不凭空添加日期边界 |
| QA 协议 | 最后一行以 `So the answer is:` 开头，后接实际答案；实体与年份／月份／日期按问题要求输出 |
| GP 与 QA 请求 | `temperature=0`、`max_tokens=4096`、客户端并发 8 |
| 时间处理 | 有年份、完整日期及部分带时分秒日期的兼容处理；不是任意自然语言日期解析器 |
| 目录调整 | 共享索引／向量、分模型结果和启动脚本整理；不改变向量内容 |

Slot selection 是明确的提示词行为调整，因此本仓库结果应描述为“千问模型与所列兼容／提示词适配后的复现”，而不是“提示词完全不变、仅替换模型”。日期容错也不意味着日期比较、首次／末次推理一定正确。

图字典当前支持 `(h,r,?)`、`(?,r,t)`、`(h,?,t)`、`(h,r,t)` 四类键；不能把它理解为支持任意数量问号的通配查询。图候选为空时，原生入口会回退到文本检索事实。

QA 的事实预算为 50。当前采样函数在候选超过 50 条时，因抽样索引去重，实际保留 49 条；不超过 50 条时全部保留。这里说明现有实现，不修改其采样策略。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `grep -n 'Slot selection' datasets/MultiTQ/prompt/Qwen/gp_prompt.txt` | Qwen GP 提示词包含槽位规则 |
| `grep -n 'So the answer is' datasets/MultiTQ/prompt/Qwen/qa_prompt.txt` | Qwen QA 提示词包含最终答案协议 |
| `git diff -- datasets/MultiTQ/prompt` | 能追溯实际提示词修改，运行中不随意覆盖 |

## 7. 模型服务与启动检查

### 7.1 正式全量入口的行为

`run-14b.sh` 和 `run-32b.sh` 调用同一 [launcher.py](scripts/full-evaluation/launcher.py)。正式启动时会：

1. 取得 `artifacts/shared/locks/full-evaluation.lock`，检查本项目是否已有管线。
2. 检查双环境、CUDA、至少 3 GiB 可用磁盘、54584 题、提示词、索引、向量、所选模型与公共生成配置。
3. 创建对应模型的新运行目录；若使用 `--resume`，先校验原目录指纹。
4. 核实 8000 端口服务归属。已确认属于本项目的目标模型或另一模型，均正常停止后按当前配置重启；无法确认归属时拒绝操作，僵尸进程不作为活动服务。
5. 最多等待 600 秒，核对模型别名与权重路径，并发起一次实际生成请求。
6. 调用现有 `src/run_pipeline.py` 执行三个阶段。

正式启动会重启已确认的本项目服务；单纯 `--check` 只做只读检查。全量结束后所选 vLLM 服务保持运行。

| 服务参数 | 14B | 32B |
|---|---:|---:|
| served-model-name | `qwen14b` | `qwen32b-gp` |
| dtype | half | half |
| max-model-len | 8192 | 8192 |
| gpu-memory-utilization | 0.80 | 0.80 |
| max-num-seqs | 64 | 8 |
| max-num-batched-tokens | 未显式指定，由该版本服务决定 | 2048 |
| generation-config | `artifacts/shared/generation-config` | 同左 |

`max-num-seqs=64` 是服务容量上限，不表示 14B 客户端并发为 64；两种模型的实验请求并发均为 8。公共生成配置里的默认 `temperature=0.7` 不会覆盖 GP／QA 显式传入的 0。

### 7.2 仅运行小样本时的手动 32B 服务

全量启动器已负责服务管理，通常直接使用第 1 节即可。以下命令仅供第 8 节直接 CLI 的新环境使用：先完成索引预建，并确认当前没有占用 8000 端口的服务或并行实验。已有正确 32B 服务且生成正常时跳过启动。

```bash
cd "$HOME/SSR"
mkdir -p artifacts/qwen2.5-32b-awq/service
VLLM_LOG="$PWD/artifacts/qwen2.5-32b-awq/service/vllm-manual-$(date +%Y%m%d-%H%M%S).log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
nohup "$HOME/miniconda3/envs/vllm/bin/vllm" serve "$PWD/models/qwen32b-awq" \
  --served-model-name qwen32b-gp --host 127.0.0.1 --port 8000 \
  --dtype half --max-model-len 8192 --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.80 \
  --generation-config "$PWD/artifacts/shared/generation-config" \
  --disable-log-requests > "$VLLM_LOG" 2>&1 < /dev/null &
printf 'vLLM日志：%s\n' "$VLLM_LOG"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定模型和日志路径 |
| 2 | `mkdir -p artifacts/qwen2.5-32b-awq/service` | 准备该模型的服务日志目录 |
| 3 | `VLLM_LOG=...` | 为本次服务创建独立日志名 |
| 4 | `CUDA_VISIBLE_DEVICES=0 ... nohup ... serve ... &` | 按 32B 参数后台启动本地服务 |
| 5 | `printf ... "$VLLM_LOG"` | 显示本次日志位置 |

服务首次加载期间可能尚未监听端口。查看刚打印的日志，确认正常加载后再检查接口；不要连续启动多个实例。

```bash
curl --max-time 10 -fsS http://127.0.0.1:8000/v1/models
"$HOME/miniconda3/envs/ssr/bin/python" - <<'PY'
import json
from urllib.request import Request, urlopen
payload = {'model': 'qwen32b-gp', 'temperature': 0, 'max_tokens': 8,
           'messages': [{'role': 'user', 'content': 'Reply OK.'}]}
request = Request('http://127.0.0.1:8000/v1/chat/completions',
                  data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
with urlopen(request, timeout=60) as response:
    result = json.load(response)
print(result['choices'][0]['message']['content'])
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `curl .../v1/models` | 查看服务别名与模型根路径 |
| 2 | `.../ssr/bin/python - <<'PY' ...` | 确认实际生成可用，而不只检查模型列表 |

### 7.3 停止实验与 vLLM 释放显存

全量任务退出后，独立运行的 vLLM 仍可能占用显存。释放这部分显存时，先停止仍在运行的实验，再停止模型服务。

**第一步：若实验仍在运行，进入对应会话停止。** 没有运行中的实验时跳过。

```bash
tmux attach -t ssr-32b-full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux attach -t ssr-32b-full` | 进入 32B 实验；14B 使用 `ssr-14b-full` |

会话内按 **Ctrl+C**，等待任务退出，再按 **Ctrl+b**、**d** 返回服务器终端。只停止实验时，到这里即可；需要释放模型服务占用的显存时继续下一步。

**第二步：停止本项目的 14B／32B 服务。** 以下命令复用当前启动器的进程检查和停止逻辑，自动识别这两种模型。仍有实验运行时会拒绝停止；无法确认归属的服务会保留并报告原因。

```bash
cd "$HOME/SSR"
"$HOME/miniconda3/envs/ssr/bin/python" - <<'PY'
import fcntl
import importlib.util
from pathlib import Path

root = Path.cwd().resolve()
spec = importlib.util.spec_from_file_location(
    'ssr_launcher', root / 'scripts/full-evaluation/launcher.py'
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

lock_path = root / 'artifacts/shared/locks/full-evaluation.lock'
lock_path.parent.mkdir(parents=True, exist_ok=True)

with lock_path.open('a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('实验仍在运行，请先进入对应tmux会话按Ctrl+C停止。')

    launcher.check_processes(root)
    state = launcher._inspect(root, launcher.PROFILES['32'])

    if state['state'] in ('target', 'other_ssr'):
        print('正在停止：', state['model_root'], flush=True)
        launcher._stop(state['_members'], state['pid'], print)
        print('本项目vLLM已停止。')
    elif state['state'] == 'stopped':
        print('本项目vLLM已处于停止状态。')
    else:
        raise SystemExit('未停止未知进程：' + state.get('reason', str(state)))
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定启动器及运行锁的位置 |
| 2 | `.../ssr/bin/python - <<'PY' ...` | 持有运行锁，检查实验和服务归属，再正常停止已确认的模型进程树 |

`PROFILES['32']` 在这里用于观察服务；随后同时接受 `target` 和 `other_ssr`，因此在线的是本项目 14B 时也会正确识别并停止。停止会等待相关进程退出，超时会报告残留进程。GitHub 同步器由独立的 `sync` 会话管理，见第 12.2.5.3 节。

**第三步：确认显存和端口状态。**

```bash
nvidia-smi
ss -ltnp 'sport = :8000'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `nvidia-smi` | 查看模型进程是否退出，以及 Memory-Usage 是否下降 |
| 2 | `ss -ltnp 'sport = :8000'` | 确认原模型服务已不再监听 8000 端口 |

显存可能需要几秒释放；如果还有其他任务占用 GPU，总占用不会归零。若本项目进程仍在或显存没有预期下降，保留命令输出，核对具体 PID 与归属后继续定位。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `/v1/models` 查询 | 启动后别名及 root 与所选本地权重对应 |
| 最小生成请求 | 启动后返回非空实际回答 |
| 本次服务日志 | 没有启动异常；完整实验进度另看运行日志 |
| 停止后的 `nvidia-smi`、`ss` | 本项目模型进程退出，显存下降，原服务不再监听 8000 端口 |

## 8. 小样本验证与原生 CLI

原生入口是 `src/run_pipeline.py`。三个阶段开关为 `store_true`，直接写开关即可，不附加 `True` 或 `False`。不显式启用 QA 时，不能把只完成检索的运行当作完整问答评测。

以下用当前 32B 服务跑 seed 42 的 20 题，并为本次检查创建独立缓存。运行前应已准备四份资产和正确的模型服务。

```bash
cd "$HOME/SSR"
RUN_DIR="artifacts/qwen2.5-32b-awq/runs/smoke20-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
set -o pipefail
"$HOME/miniconda3/envs/ssr/bin/python" -u src/run_pipeline.py \
  --run-text-retrieval --run-graph-retrieval --run-graph-qa \
  --text-retriever-model-name "$PWD/models/bge-m3" --device cuda:0 \
  --text-index-path artifacts/shared/indexes/multitq_text_index.pkl \
  --graph-cache-path artifacts/shared/indexes/multitq_graph_retriever.pkl \
  --base-url http://127.0.0.1:8000/v1 --api-key dummy \
  --llm-model-name qwen32b-gp --concurrency 8 --random-seed 42 \
  --text-top-k 100 --qa-retrieve-cnt 50 \
  --graph-pattern-prompt datasets/MultiTQ/prompt/Qwen/gp_prompt.txt \
  --qa-prompt datasets/MultiTQ/prompt/Qwen/qa_prompt.txt \
  --cache-path "$RUN_DIR/llm_cache.jsonl" --sample-size 20 \
  2>&1 | tee "$RUN_DIR/run.log"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定数据与提示词路径 |
| 2 | `RUN_DIR=...` | 新建 32B 小样本运行名 |
| 3 | `mkdir -p "$RUN_DIR"` | 创建运行目录 |
| 4 | `export OMP_NUM_THREADS=8 ...` | 设置 CPU 线程 |
| 5 | `set -o pipefail` | 使 Python 失败能穿过 tee 被检测 |
| 6 | `.../ssr/bin/python -u src/run_pipeline.py ...` | 完整运行 20 题，保存日志和独立请求缓存 |

改为 14B 时，应先准备 14B 服务，将模型别名改为 `qwen14b`，运行目录归入 `artifacts/qwen2.5-14b-awq/runs/`。不要仅修改目录名而继续请求另一种模型。

原生全量 CLI 省略 `--sample-size`，同时使用新的独立缓存。长期全量任务优先使用第 1 节已核对的 tmux 启动器。普通实验不加 `--force-reload`；它会触发索引重建，不属于正常续跑步骤。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `tail -n 12 "$RUN_DIR/run.log"` | 题数与所选样本一致，有 QA 指标和 `Done` |
| `/v1/models` 查询 | 实际调用的是所选模型 |
| 本次缓存路径 | 与其他模型、旧提示词运行分开，不设无依据的准确率门槛 |

## 9. 缓存续跑与完成判据

### 9.1 每次全量运行保存什么

```text
artifacts/qwen2.5-32b-awq/runs/full-年月日-时分秒-微秒/
├── run.json             # 配置、文件指纹、服务信息、状态、退出码、总分
├── run.log              # 原生管线输出；续跑时追加
└── llm_cache.jsonl      # GP 与 QA 请求缓存
```

14B 在对应模型目录采用相同结构。准备服务失败时可能只有 `run.json`，不要把目录存在或 tmux 存活当作任务完成。正常完成以 `run.json` 的 `status=completed` 为准，并核对本次日志。

查看两个模型最近一次**本启动器创建的全量运行**，以下命令只读文件，不请求模型：

```bash
cd "$HOME/SSR"
"$HOME/miniconda3/envs/ssr/bin/python" - <<'PY'
import json
from pathlib import Path
for model in ('qwen2.5-14b-awq', 'qwen2.5-32b-awq'):
    states = sorted((Path('artifacts') / model / 'runs').glob('full-*/run.json'))
    if not states:
        print(model, '未找到新启动器的全量运行记录')
        continue
    path = states[-1]
    row = json.loads(path.read_text(encoding='utf-8'))
    print(model, path.parent, row.get('status'), row.get('metrics'), row.get('error'))
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 定位本项目运行目录 |
| 2 | `.../ssr/bin/python - <<'PY' ...` | 显示两模型最近一次新全量运行状态；历史成绩另见第 0 节 |

### 9.2 同配置续跑

将下面 `full-实际目录名` 替换成会话中打印的、由本启动器创建的未完成目录。它必须直接位于对应模型的 `runs/` 下；不要指向 `history/` 或普通 CLI 的 smoke 目录。

```bash
cd "$HOME/SSR"
bash scripts/full-evaluation/run-32b.sh --resume "$HOME/SSR/artifacts/qwen2.5-32b-awq/runs/full-实际目录名"
tmux attach -t ssr-32b-full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定续跑项目 |
| 2 | `bash scripts/full-evaluation/run-32b.sh --resume ...` | 校验原运行指纹，保留缓存后重新进入管线 |
| 3 | `tmux attach -t ssr-32b-full` | 查看恢复过程 |

14B 使用 `run-14b.sh`、14B 的原运行目录及 `ssr-14b-full`。续跑会重新进行管线准备和检索，并复用有效的模型请求缓存；它不是旧冻结脚本按 `row_index` 从某一行继续执行。

启动器比较代码、启动脚本、中央提示词、数据、配置、模型与环境等指纹。大型权重、索引和 NPY 使用文件大小／修改时间等状态，另有部分结构校验，不是全部资产逐字节哈希。变化后应新建运行；不修改 `run.json` 来绕过检查。已标记 `completed` 的目录不会被当作续跑目标。

**两份向量迁入共享目录的改动也改变了源码和资产路径。迁移前创建的 `run.json` 不再满足新指纹；历史完成结果保留，新实验使用当前入口创建新目录。**

### 9.3 缓存与评分口径

普通请求缓存键为 `prompt、model、max_tokens、stop、logprobs`，不包含接口地址、权重内容、KG 或索引指纹。当前代码在温度为 0 时复用符合条件的缓存，含 `error` 的回答不会作为有效缓存返回。新实验独立缓存、续跑额外核对资产，分别解决缓存归属与配置一致性问题。

同一道问题可能复用请求，失败重试也会改变缓存记录数。**缓存行数不能代替测试集 54584 条原始记录，缓存命中速度也不能作为实时推理吞吐。**

当前 [eval_utils.py](src/utils/eval_utils.py) 的 `Graph QA Accuracy` 按题计算：预测若为列表则先取第一项，再与标准答案列表中的任一项做精确字符串比较；不是完整实体集合匹配。评测不自动统一大小写、别名或年月日粒度。`2010`、`2010-01`、`2010-01-15` 是不同字符串。

QA 提示词要求精确最终标记；解析器实际寻找大小写敏感的 `So the answer is`，取最后一次出现之后的文本，清理冒号和末尾句号。缺少标记的回答返回 `ERROR`。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| 本次 `run.json` | `status=completed`，`exit_code=0`，`metrics.total=54584` |
| 本次 `run.log` | 同一轮有 `Loaded 54584 questions`、分母 54584 的 Graph QA 指标以及 `Done` |
| 续跑前的指纹校验 | 配置和资产匹配，缓存 JSONL 可解析 |

## 10. 历史归档与日常维护

### 历史与当前运行

`history/` 保存原实验的摘要、原始记录或日志。32B 的 `metadata/historical-manifest.json` 包含迁移前路径和指纹，仅用于历史追溯。历史 `.disabled` 脚本、旧完成标记不能用作新启动器的运行入口或续跑清单。

旧的根目录安装文件 `create_full_evaluation.sh` 若仍存在，嵌入的是迁移前启动器模板。正式维护入口是当前仓库的 `scripts/full-evaluation/`，无需再次使用旧安装模板。

### 文件保留与清理

| 类型 | 用途 | 处理建议 |
|---|---|---|
| `datasets/`、`src/`、`scripts/` | 运行输入和程序 | 保留当前生效版本 |
| `models/` | 本地模型权重 | 保留正在使用的模型 |
| `artifacts/shared/indexes/` 四份资产 | 加载检索器、编码对齐 | 来源一致时复用；缺失才预建 |
| `artifacts/shared/generation-config/` | 公共生成默认配置 | 随实验记录管理，运行中不改 |
| `runs/full-*/llm_cache.jsonl` | 模型响应复用与离线核查 | 需要续跑或核查时保留 |
| `runs/full-*/run.json`、`run.log` | 配置、状态、指标与失败定位 | 保留有效实验记录 |
| `service/*.log` | 对应 vLLM 实例的启动和吞吐信息 | 确认实例已退出、记录不再需要后单独整理 |
| `history/` | 已完成结果和对照证据 | 先确定保留范围，再单独清理 |
| `__pycache__/` | Python 字节码缓存 | 可再生；与实验数据分开处理 |

清理是独立操作：先停止相关管线、核对共享节点没有共写，再限定具体目录和文件白名单。不要把 `artifacts/` 整体加入删除范围，也不要把清缓存夹在正常启动或续跑命令中。此 README 不执行删除。

### Git 跟踪边界

当前 [.gitignore](.gitignore) 保持两份 NPY 受跟踪，同时忽略同目录的两个大型 PKL；模型、请求缓存、运行锁和同步器自身日志也被忽略。普通实验／服务日志可被同步。GitHub 中没有 ignored 文件不能证明服务器没有这些产物。

```bash
cd "$HOME/SSR"
git ls-files artifacts/shared/indexes
git check-ignore -v artifacts/shared/indexes/multitq_text_index.pkl artifacts/shared/indexes/multitq_graph_retriever.pkl
git status --short
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 在正确仓库检查规则 |
| 2 | `git ls-files artifacts/shared/indexes` | 确认两份 NPY 的版本跟踪状态 |
| 3 | `git check-ignore -v ...两个PKL` | 确认大型索引被忽略 |
| 4 | `git status --short` | 查看实际改动；不把自动同步等同于 ignored 资产备份 |

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `git ls-files artifacts/shared/indexes` | 包含两份 NPY |
| 两个 PKL 的 `git check-ignore` | 匹配共享索引忽略规则 |
| 历史目录中的摘要和日志 | 与新 `runs/` 的状态文件分开使用 |

## 11. 常见问题

| 现象 | 含义或待确认原因 | 处理方式 |
|---|---|---|
| 找不到 `run-14b.sh` / `run-32b.sh` | 仓库版本较旧、目录错误或脚本尚未放到服务器 | 核对服务器上的 `scripts/full-evaluation/` 和实际代码版本 |
| `Set-Clipboard` 没有输出 | Windows 只完成了本机复制动作 | 在服务器执行实际脚本；也可上传文件后用 Bash 执行 |
| 数据路径不存在 | 没从项目根启动，或输入未准备 | 先进入 `$HOME/SSR`，再执行第 4 节检查 |
| `unrecognized arguments: False` | 阶段开关后附加了布尔值 | 需要阶段就写开关，不需要则省略 |
| 预检提示缺少 PKL 或 NPY | 启动器要求先准备四份检索资产 | 按第 5 节单独预建，不让 Qwen 与完整建库争显存 |
| 向量行数、维度或 dtype 异常 | ID 映射、BGE 或向量来源不一致 | 核对资产来源，再决定是否重建；不跳过校验 |
| `other_ssr` | 在线的是本项目另一种模型 | 属于可识别状态，正式启动时按所选模型切换 |
| `blocked` / 未就绪或无法归属的进程 | 服务尚未就绪或归属无法确认 | 核对端口、当前用户进程及日志，先排除重复实例 |
| `[vllm] <defunct>` | 僵尸进程 | 结合端口和 GPU 判断，不能仅因名称存在就重复终止或启动 |
| `Connection refused` | 服务尚未监听、启动失败或位于另一节点 | 查看本次服务日志和节点；不要连开多个实例 |
| 有模型列表但生成失败 | 列表接口不证明推理正常 | 做实际生成检查，检查服务日志 |
| 实验退出后显存仍被模型占用 | vLLM 是独立的常驻服务 | 按第 7.3 节停止模型服务，再核对 GPU 和端口 |
| `CUDA out of memory` | 可能存在建库、模型服务或其他任务争用 | 看现场进程和显存；首次建库先释放相关模型服务 |
| 32B 很慢且 GPU 100% | 单卡计算和长输入处理可能饱和 | 联合查看 prompt/generation throughput、请求数和 KV 使用，不单凭显存空闲判断应加并发 |
| 进度停在 `197/200` | 还剩请求未完成，可能有长生成或重试 | 197 是完成数，不是题目编号；比较日志与 token 增量 |
| GP 很快完成 | 可能正常命中请求缓存 | 核对缓存归属，不将其当作实时吞吐 |
| 有自然语言回答但预测为 `ERROR` | 缺少规定的最终答案标记 | 保留原回答核查协议，不能直接认定模型本来答对 |
| `--resume` 报资产变化 | 代码、提示词、目录或权重状态与原记录不同 | 保留原记录，新建实验；不改指纹绕过检查 |
| 缓存某行损坏 | JSONL 不完整或内容异常 | 先保留文件并定位具体行；不要直接清空全部缓存 |
| 提示已有运行锁或已有会话 | 可能有本项目任务运行 | 进入原会话查看；不要靠删除锁文件启动第二份 |
| tmux 留下已退出窗格 | 启动器保留退出结果，供检查失败或完成信息 | 查看日志与 `run.json`；再次启动会处理全部已退出的同名会话 |
| 历史 Single / Multiple 为 null | 当前读取的分组字段与数据字段可能不同 | 与 Overall、Entity、Time 分开理解，不擅自改变评测口径 |

## 12. tmux、同步与资料来源

### 12.1 常用会话操作

```bash
tmux list-sessions
tmux attach -t ssr-32b-full
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux list-sessions` | 查看当前节点会话 |
| 2 | `tmux attach -t ssr-32b-full` | 查看 32B；14B 对应 `ssr-14b-full` |

会话内 Ctrl+b、d 脱离；Ctrl+C 中断前台任务。模型服务是独立进程，管线退出不等于模型服务退出。全量启动器使用 `remain-on-exit` 保留结束画面，最终完成仍以第 9 节判据为准。

### 12.2 GitHub 自动同步完整流程

以下包含同步的配置、启动、验证、恢复和排障流程；独立文件 [SSR_GitHub自动同步.md](SSR_GitHub自动同步.md) 保留同样的实用内容，便于单独查阅。

#### 12.2.0. 使用方式与同步方向

```text
服务器 $HOME/SSR
  └─ tmux 会话 sync
       └─ scripts/github-sync/autosync.sh
            ├─ git add -A：收集新增、修改和删除
            ├─ 检查本轮暂存文件的大小
            ├─ 有暂存变化时创建 auto 提交
            └─ 有待推标记时推送当前 HEAD → backup/main

同步器日志：scripts/github-sync/autosync.log（不跟踪）
```

已配置的服务器先读第 12.2.1、12.2.5、12.2.6 节；首次配置再执行第 12.2.2 节。以下命令均在服务器终端运行，仓库根与运行根统一为 `$HOME/SSR`。

这是**服务器向 GitHub 推送**的机制。脚本不从 GitHub 自动拉取、合并或检出文件；在网页或另一台机器修改同一分支后，需要按第 12.2.7 节处理分歧。

#### 12.2.1. 已有环境先检查

```bash
cd "$HOME/SSR"
git rev-parse --show-toplevel
git branch --show-current
git remote -v
tmux list-sessions 2>&1 || true
pgrep -u "$(id -u)" -af '[s]cripts/github-sync/autosync.sh' || true
tail -n 20 scripts/github-sync/autosync.log
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入实际同步的仓库 |
| 2 | `git rev-parse --show-toplevel` | 确认仓库根与运行根一致 |
| 3 | `git branch --show-current` | 确认当前分支；本项目约定为 main |
| 4 | `git remote -v` | 核对 backup 指向自己的 SSR 仓库，不假定 origin 的地址 |
| 5 | `tmux list-sessions` | 查看当前节点是否已有 sync 会话 |
| 6 | `pgrep -u ... -af ...` | 查看当前用户的同步进程；包装 shell 与脚本可能分别显示 |
| 7 | `tail -n 20 .../autosync.log` | 查看实际提交、推送或错误；尚未启动过时可能没有此文件 |

已有同步进程正常工作时继续使用，不重复生成密钥、添加 remote 或启动实例。会话存在不能单独证明推送成功，还需第 12.2.6 节的提交与远端核对。当前脚本没有单实例锁，同一节点及共享工作区应只运行一套同步器。

#### 12.2.2. 首次配置权限与远端

##### 12.2.2.1. 准备专用密钥

已有 `$HOME/.ssh/github_ssr` 时优先复用。下面的生成命令仅在私钥、公钥都不存在时执行；如果只剩其中一个文件，先核对原配置，不覆盖已有密钥。

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [ ! -e "$HOME/.ssh/github_ssr" ] && [ ! -e "$HOME/.ssh/github_ssr.pub" ]; then ssh-keygen -t ed25519 -f "$HOME/.ssh/github_ssr" -N '' -C "ssr-autosync@$(hostname -s)"; else printf '密钥文件已存在，请核对并复用。\n'; fi
chmod 600 "$HOME/.ssh/github_ssr"
cat "$HOME/.ssh/github_ssr.pub"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `mkdir -p "$HOME/.ssh"` | 准备 SSH 目录 |
| 2 | `chmod 700 "$HOME/.ssh"` | 限制目录访问权限 |
| 3 | `if ...; then ssh-keygen ...; else ...; fi` | 仅在两个文件均不存在时生成无口令的专用密钥 |
| 4 | `chmod 600 .../github_ssr` | 设置私钥权限 |
| 5 | `cat .../github_ssr.pub` | 显示公钥，供添加到 GitHub；私钥不放入仓库或文档 |

在 GitHub 仓库的 **Settings → Deploy keys → Add deploy key** 中添加公钥，并启用 **Allow write access**。Deploy key 默认只读，自动推送需要写权限；具体配置见 [GitHub 官方说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)。

##### 12.2.2.2. 首次连接与读取检查

首次 SSH 连接出现主机指纹确认时，先与 [GitHub 官方 SSH 指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)核对，再确认连接。已有正确 known_hosts 时沿用，不盲目追加扫描到的主机密钥。

```bash
ssh -T -i "$HOME/.ssh/github_ssr" -o IdentitiesOnly=yes git@github.com
cd "$HOME/SSR"
git remote get-url backup || git remote add backup git@github.com:2200170080/SSR.git
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_ssr -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15" \
  git ls-remote backup
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh -T -i ... git@github.com` | 核对专用密钥认证；GitHub 不提供 shell，认证成功后返回码仍可能为 1 |
| 2 | `cd "$HOME/SSR"` | 进入要推送的仓库 |
| 3 | `git remote get-url backup ...` | 已有则显示地址，缺失才新增 backup；已有错误地址时先人工核对 |
| 4 | `GIT_SSH_COMMAND=... git ls-remote backup` | 验证该密钥能读取目标仓库引用 |

`ssh -T` 证明身份认证，`git ls-remote` 证明读取；二者都不能单独证明写权限或绕过分支保护。真实推送成功和第 12.2.6 节核对才构成推送证据。`origin` 可以指向自己的仓库或其他来源，以实际输出为准；自动脚本默认推送的是 `backup`。

同步器会显式指定密钥，无需为了本脚本覆盖全局 `~/.ssh/config`。自动提交身份由脚本的 `AUTHOR_NAME`、`AUTHOR_EMAIL` 控制，不要求把旧文档中的邮箱写进全局 Git 配置。

#### 12.2.3. 哪些文件同步

| 文件类别 | 当前处理 | 说明 |
|---|---|---|
| README、同步说明、src、scripts | 跟踪并同步 | 包括本次更新的两份 Markdown |
| `datasets/MultiTQ/` | 跟踪并同步 | KG、ID、测试问题、生效提示词；当前没有整体忽略 datasets |
| 两份 `artifacts/shared/indexes/MultiTQ_*_embeddings.npy` | 显式放行并跟踪 | 两份向量均低于默认单文件阈值 |
| `artifacts/shared/indexes/*.pkl` | 忽略 | 大型文本索引和图缓存，保留在服务器 |
| `models/`、`.hf_cache/`、上游 `MultiTQ/` 下载目录 | 忽略 | 权重与下载缓存不作为代码同步内容 |
| `artifacts/**/llm_cache.jsonl`、`cache.jsonl` | 忽略 | 请求缓存单独保留和迁移 |
| `artifacts/shared/locks/`、`run.json.tmp` | 忽略 | 运行锁与临时状态 |
| `artifacts/**/*.log` | 通常放行 | 仍受单文件大小守卫及目录局部忽略规则影响 |
| `scripts/github-sync/autosync.log` | 忽略 | 避免同步器自身输出形成重复提交链 |
| 历史目录的 `bulk/` 等 | 按对应局部 `.gitignore` | GitHub 不一定包含完整实验原始产物 |

当前共享索引规则的要点如下，这是说明性摘录；已有正确规则无需重复追加：

```gitignore
/artifacts/shared/indexes/*
!/artifacts/shared/indexes/MultiTQ_entity_embeddings.npy
!/artifacts/shared/indexes/MultiTQ_relation_embeddings.npy
/artifacts/**/llm_cache.jsonl
/artifacts/**/cache.jsonl
/artifacts/shared/locks/
/artifacts/**/run.json.tmp
!/artifacts/**/*.log
/scripts/github-sync/autosync.log
```

核对规则与跟踪状态：

```bash
cd "$HOME/SSR"
git ls-files -- datasets/MultiTQ scripts/github-sync/autosync.sh artifacts/shared/indexes
git check-ignore --no-index -v artifacts/shared/indexes/multitq_text_index.pkl scripts/github-sync/autosync.log
git check-ignore --no-index -v artifacts/shared/indexes/MultiTQ_entity_embeddings.npy artifacts/shared/indexes/MultiTQ_relation_embeddings.npy
cat .git/info/exclude
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定检查的仓库 |
| 2 | `git ls-files -- ...` | 查看实际已跟踪的输入、脚本和两份向量 |
| 3 | `git check-ignore --no-index -v ...PKL ...autosync.log` | 查看应被忽略文件的命中规则 |
| 4 | `git check-ignore --no-index -v ...两份NPY` | 查看向量的精确放行规则 |
| 5 | `cat .git/info/exclude` | 查看仅本机生效的额外规则；不写入新的兜底规则 |

带 `!` 的匹配规则表示放行，不能把 `git check-ignore -v` 的所有输出都理解为“被忽略”。单文件用 `--no-index -q` 检查时，返回码 0 表示被忽略，1 表示未被忽略，其他返回码需排查错误。已跟踪文件不会仅因新增 ignore 规则就自动停止跟踪；用 `git ls-files` 同时核对。规则解释参考 [Git 官方文档](https://git-scm.com/docs/git-check-ignore)。

#### 12.2.4. 脚本参数与真实行为

| 变量 | 默认值或来源 | 作用 |
|---|---|---|
| `ROOT` | `$HOME/SSR` | 同步的仓库根 |
| `REMOTE` | `backup` | 推送远程名 |
| `BRANCH` | `main` | 远端目标分支 |
| `INTERVAL` | `60` | 每轮结束后的等待秒数 |
| `MAX_MB` | `50` | 文件阈值；实际按 `50 × 1024 × 1024` 字节，即 50 MiB 判断 |
| `RETRIES` | `3` | 本轮推送尝试次数，失败尝试间隔 15 秒 |
| `AUTHOR_NAME` | `2200170080` | 自动提交作者名，可用环境变量覆盖 |
| `AUTHOR_EMAIL` | 脚本中的提交邮箱 | 可覆盖为本人提交邮箱；自动 commit 使用命令级 Git 配置 |

每轮先执行 `git add -A`，因此受跟踪文件的**删除和改名也会被提交**。随后检查本轮暂存列表：若对应工作文件仍存在且大于阈值，脚本用按路径的 `git reset` 将其撤出暂存区，保留工作文件；其余暂存变化才生成 `auto: 时间 @ 主机名` 提交。

大小守卫不是完整历史清理：它不会删除工作区大文件，不会移除已经进入提交历史的大 blob，也不会阻止已跟踪大文件的删除记录被提交。日志被放行也可能因超过阈值而停止更新。不要把“当前文件已删除”理解为“旧大对象已从待推历史中消失”。

有新提交时，内存变量 `NEED_PUSH` 置为 1，脚本推送**当前本地 HEAD** 到指定远程分支；它不会自动切换本地分支。推送失败后，同一进程会在下一轮继续尝试。每轮还有 Git 操作、网络和重试耗时，不能保证严格每 60 秒完成一次推送。

**待推标记不持久化。** 脚本每次启动都把 `NEED_PUSH` 初始化为 0；若重启时本地已经领先远端，但没有新的成功自动提交，或者提交由外部手动创建，脚本不保证自动补推这些提交。遇到这种情况按第 12.2.7 节核对并补推已有提交。

脚本会尝试将 `$HOME/.ssh/github_ssr` 复制到当前节点的 `/tmp/.autosync_gh_key`，设置私钥权限为 600，并通过 `/tmp/.autosync_ssh/` 的连接复用减少重复握手。`/tmp` 是节点本地运行资料，不作为 GitHub 同步内容；连接复用不能保证消除所有认证或网络错误。

#### 12.2.5. 启动、查看、停止与重启

##### 12.2.5.1. 启动一套同步器

仅在第 12.2.1 节确认本项目没有同步进程时执行。以下固定使用 `backup/main`；有意使用其他目标时应同时调整后续验证命令。

```bash
cd "$HOME/SSR"
mkdir -p scripts/github-sync
tmux has-session -t '=sync' 2>/dev/null || \
  tmux new-session -d -s sync -c "$HOME/SSR" \
    bash -c 'set -o pipefail; ROOT="$HOME/SSR" REMOTE=backup BRANCH=main INTERVAL=60 MAX_MB=50 RETRIES=3 bash "$HOME/SSR/scripts/github-sync/autosync.sh" 2>&1 | tee -a "$HOME/SSR/scripts/github-sync/autosync.log"'
tmux list-sessions
tail -n 20 scripts/github-sync/autosync.log
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入同步仓库 |
| 2 | `mkdir -p scripts/github-sync` | 确保日志目录存在，脚本应已随仓库准备好 |
| 3 | `tmux has-session ... \|\| tmux new-session ...` | 同名会话不存在时才创建；传入明确参数，将输出追加到同步器日志 |
| 4 | `tmux list-sessions` | 确认会话是否仍在 |
| 5 | `tail -n 20 .../autosync.log` | 查看初始化、提交或推送错误；创建会话不等于推送成功 |

这个会话检查不能代替进程检查；如果已有在别的会话或前台运行的同步器，应继续使用或先正常停止它，不再启动第二份。

##### 12.2.5.2. 查看与脱离

```bash
tmux attach -t sync
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux attach -t sync` | 查看实时输出；按 Ctrl+b，再按 d 脱离并保持后台运行 |

##### 12.2.5.3. 暂停与恢复

确认 `sync` 就是本项目的同步会话后，以下只停止同步器；实验会话 `ssr-14b-full`、`ssr-32b-full` 分别管理。

```bash
tmux kill-session -t '=sync'
pgrep -u "$(id -u)" -af '[s]cripts/github-sync/autosync.sh' || true
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux kill-session -t '=sync'` | 结束已确认的同步会话 |
| 2 | `pgrep -u ... -af ...` | 确认同步脚本是否仍有残留；有残留时先核对，不立即重启 |

确认同步及相关 Git 子进程已退出后，再处理远端分歧或本地提交；恢复时重新执行第 12.2.5.1 节。重启节点后，tmux 本身不自动恢复会话，先检查是否已有其他恢复机制，再决定是否手动启动。

#### 12.2.6. 只读验证同步结果

等本轮提交和推送处理结束后执行。使用同步器的节点本地密钥副本查询，与正在运行的脚本保持一致：

```bash
cd "$HOME/SSR"
set -o pipefail
git status --short
git log -1 --format='%h %s'
git rev-parse HEAD
GIT_SSH_COMMAND='ssh -i /tmp/.autosync_gh_key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15' \
  git ls-remote --exit-code backup refs/heads/main
tail -n 20 scripts/github-sync/autosync.log
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 固定核对仓库 |
| 2 | `set -o pipefail` | 为当前 shell 启用管道失败传播 |
| 3 | `git status --short` | 查看是否还有未提交变化或被守卫跳过的文件 |
| 4 | `git log -1 ...` | 查看最新实际提交 |
| 5 | `git rev-parse HEAD` | 显示本地完整提交 SHA |
| 6 | `GIT_SSH_COMMAND=... git ls-remote --exit-code ...` | 只读查询远端 main 的完整 SHA；同时确认查询没有报错 |
| 7 | `tail -n 20 .../autosync.log` | 结合 pushed 或失败信息判断本轮结果 |

远端输出第一列 SHA 与本地 HEAD 相同，表示远端分支指向同一提交；还应核对预期文件已经提交，而不是仅比较两个旧提交。如果恰好遇到同步器创建新提交，可能出现短暂不一致，待本轮完成后再查。新空仓库没有 `main` 时，`ls-remote --exit-code` 会报告找不到该引用。

这些检查不会向 README 追加测试行。工作区没有待提交变化、日志暂时安静，可能只是没有新内容；也可能是改动被忽略、超过阈值、提交失败或存在待补推提交，需要联合核对。远端 SHA 相等不能证明模型、PKL、响应缓存和历史 bulk 已备份。

#### 12.2.7. 补推已有提交与处理分歧

先按第 12.2.5.3 节停止同步，确认没有并行 Git 写操作。以下检查使用 HOME 中的原专用密钥，避免依赖节点重启后可能不存在的 `/tmp` 副本：

```bash
cd "$HOME/SSR"
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_ssr -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
git fetch backup main
git status --short
git log --oneline --left-right HEAD...FETCH_HEAD
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/SSR"` | 进入待检查仓库 |
| 2 | `export GIT_SSH_COMMAND=...` | 为当前终端的后续 Git 连接指定专用密钥 |
| 3 | `git fetch backup main` | 获取远端引用，不合并或覆盖工作文件 |
| 4 | `git status --short` | 查看需要先保存或确认的本地变化 |
| 5 | `git log --oneline --left-right HEAD...FETCH_HEAD` | `<` 表示仅本地有的提交，`>` 表示仅远端有的提交 |

只有 `<` 时，检查这些本地提交确实需要推送，先预演：

```bash
git push --dry-run backup HEAD:refs/heads/main
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git push --dry-run ...` | 检查推送目标和预期更新，不实际更新远端；不能代替真实写入结果 |

预演结果符合预期后，再推送现有本地提交：

```bash
git push backup HEAD:refs/heads/main
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git push backup HEAD:refs/heads/main` | 补推已经存在的本地提交，不重新暂存全部文件 |

如果存在 `>` 或两边都有提交，先核对网页／另一节点的变更及本地工作文件，再按具体情况合并或 rebase、处理冲突。脚本没有自动处理这类分歧的能力。不要把强制推送、重写历史或删除大文件历史作为日常同步步骤。

补推成功后按第 12.2.6 节复核，再恢复同步器。推送因旧大对象被拒时，暂停并定位具体对象，单独制定历史整理方案；只增加 ignore 或删除当前文件不能自动解决历史对象问题。

#### 12.2.8. 换机与版本保存

代码、文档和被跟踪输入可从 GitHub 获取；模型、PKL、请求缓存及被忽略的实验原始文件需要单独迁移。两份 NPY 已受跟踪，但仍应核对新位置和实际文件内容。

共享 HOME 不共享节点的 GPU、tmux 和 `/tmp`。换节点后核对专用密钥、backup 地址、当前分支和单实例状态；不让两个节点同时对同一个共享工作区运行同步器或共写同一实验目录。

Git 标签保存受跟踪版本。它不会自动保存 ignored 资产，也不会自动把工作区恢复到标签。需要恢复历史代码时，先独立核对版本和运行资料，再处理当前工作区与同步任务。

#### 12.2.9. 常见问题与验收

| 现象 | 原因或待核实项 | 处理 |
|---|---|---|
| 有 sync 会话但没有 pushed | 会话存在不代表认证、提交或推送成功 | 看日志、工作区状态、本地与远端 SHA |
| 没有新提交 | 可能没有暂存变化，也可能被忽略、超过阈值或提交失败 | 联合 status、ignore 和同步日志判断 |
| 重启同步器后仍未补推 | 内存待推标记已清零，且没有新自动提交 | 停同步后按第 12.2.7 节核对并补推已有提交 |
| `fetch first` / non-fast-forward | 远端有本地没有的提交 | 停同步，fetch 并检查分歧，再针对性整合 |
| `Permission denied (publickey)` | 密钥、授权、权限、连接等需要排查 | 用同一专用密钥复测，核对 Deploy key 写权限与日志，不直接认定认证限流 |
| `Host key verification failed` | known_hosts 缺失或指纹变化 | 对照 GitHub 官方指纹确认原因，避免盲目信任扫描结果 |
| `ls-remote` 成功却 push 失败 | 读取与写入权限、分支策略不同 | 核对写权限、分支保护和真实 push 报错 |
| 文件被跳过，日志含 skipped files | 本轮文件超过 MAX_MB | 判断是否应单独归档；不要直接提高阈值来绕过仓库限制 |
| 当前大文件删了，push 仍被拒 | 旧对象仍在待推历史中 | 定位被拒对象，单独处理历史，保留本地工作资料 |
| 同步器自身日志不断产生提交 | 日志被跟踪后，输出可能形成重复提交链 | 先停同步，核对 ignore 和 `git ls-files`；确认后只移除该日志的 Git 跟踪，保留服务器文件 |
| 实验日志停止更新到 GitHub | 可能超阈值、局部忽略或实验不再写入 | 看本地日志、大小与规则，再判断同步问题 |
| 文档中改了 ROOT 等参数但实例没变 | 环境变量在进程启动时读取 | 查看启动日志的实际参数，需要变更时正常重启 |
| 节点重启后没有 sync | tmux 不自动恢复，其他恢复机制未核实 | 确认无实例后按第 12.2.5 节启动 |
| `.git/index.lock` 冲突 | 可能并行 Git 操作或异常退出 | 先核对相关进程，避免与活动写入竞争，不盲目删除锁 |

| 检查项 | 验收判据 |
|---|---|
| 路径 | 脚本和日志均在 `scripts/github-sync/` |
| 实例 | 当前节点与共享工作区没有重复同步器 |
| 目标 | 当前 HEAD 的用途明确，目标远程和分支正确 |
| 规则 | 两份 NPY 与生效提示词受跟踪；PKL、模型、缓存、运行锁、同步日志被忽略 |
| 结果 | 本地与远端 SHA 对应，预期文件已提交，无未解释的跳过或失败 |

核对日期：**2026-11-07 15:24:32（Asia/Shanghai）**；默认分支 `main` 的源码基线为 [`09c70c49382239dd6b1700705f146ee03b9d4cf0`](https://github.com/2200170080/SSR/commit/09c70c49382239dd6b1700705f146ee03b9d4cf0)，依据为该提交的同步脚本、忽略规则与仓库树。服务器当前进程、密钥权限、remote 配置及推送状态未实时核对。本次仅维护说明文档，没有实际执行密钥配置、启动或推送。

### 12.3 来源、许可与核对范围

- 原论文与实现：[作者 SSR 仓库](https://github.com/zhangli-coding/SSR)。原 README 的框架图、数据来源与方法定位在此保留，并调整为本复现版本的真实入口。
- 数据：[MultiTQ](https://github.com/czy1999/MultiTQ)、[TimelineKGQA](https://github.com/PascalSun/TimelineKGQA/tree/main/Datasets)。当前完成结果属于 MultiTQ。
- 模型：[BGE-M3](https://huggingface.co/BAAI/bge-m3)、[Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)、[Qwen2.5-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)。
- 代码许可见 [LICENSE](LICENSE)；数据和权重遵循各来源方的使用条款。引用 SSR 时请使用论文正式书目信息，本 README 不补造原项目尚未给出的 BibTeX。

本 README 更新日期为 **2026-11-07**，GitHub 核对时间为 **2026-11-07 15:45:18（Asia/Shanghai）**，源码基线为默认分支 `main` 的固定提交 [`b45cd6539bfff3d6551d247e1e299a92688d88ae`](https://github.com/2200170080/SSR/commit/b45cd6539bfff3d6551d247e1e299a92688d88ae)。

文档结合现行源码、已有部署步骤、已同步历史结果和用户服务器输出整理。用户最近一次输出确认两份向量迁移成功且 14B／32B 静态预检通过；该证据不表示迁移后已经重新完成全量实验。当前服务器的实时进程、GPU、端口和工作区状态未实时核对。本文命令仅作说明，本次文档维护没有部署、启动服务或调用模型。

### 本节验收

| 检查命令 | 判据 |
|---|---|
| `git log -1 --oneline` | 能区分当前版本、本文核对基线与历史结果版本 |
| `tmux list-sessions` | 会话与本机任务对应 |
| 同步日志与 GitHub 提交 | 确认受跟踪变更已推送；不代替 ignored 资产检查 |
