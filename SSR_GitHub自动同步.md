# SSR 项目自动同步到 GitHub —— 全流程

> 适用环境：`A6000x1-3-168`（`ecs-instance-e1330fb1`），共享盘 `172.30.20.24:6789:/ltaidc`，运行根 `~/SSR`。
> 这套机制让服务器上改的代码、数据、日志、提示词每 60 秒自动 commit + push 到 `git@github.com:2200170080/SSR.git`，只有大二进制（模型、检索索引、响应缓存）不进仓库。
> 相关文档：部署手册 `SSR_部署手册.md`（第 3 节拉代码、第 10 节跑实验）；本文只讲**同步**。

---

## 0. 整体是什么



```
服务器 ~/SSR（git 仓库，仓库根 = 运行根）
        │
        │  tmux 会话 sync 里常驻 tools/autosync.sh，每 60 s：
        │    git add -A → 有变更则 commit（作者 2200170080）→ git push backup main
        ▼
GitHub 2200170080/SSR  （远端名 backup，分支 main）
```

三个特点：

1. **不需要 cron / systemd**：靠 tmux 里一个 `while true` 循环，只要有 tmux 会话就活着；
2. **不需要 conda、不需要交互**：`GIT_SSH_COMMAND` 写死专用密钥 + `BatchMode=yes`，断网/密钥错只会打印重试，不会卡在密码提示；
3. **只管「大二进制」不进去**：模型 14 GB、检索索引 1.8 GB + 115 MB、响应缓存约 700 MB 被忽略，其余全部纳入。

---

## 1. 前置条件

| 项 | 要求 | 检查命令 |
|---|---|---|
| GitHub 仓库 | 已建好、**不要**勾选自动生成 README（否则首次 push 会被拒） | 浏览器看一眼 |
| 服务器出网 | 能连 github.com、22 端口（或 443 的 ssh）可用 | `curl -s -o /dev/null -w '%{http_code}\n' https://github.com` |
| git | ≥ 2.30 | `git --version` |
| tmux | 可用（没有就 `sudo apt install tmux`，或改用 nohup） | `tmux -V` |

---

## 2. 一次性配置

> 只在**新机器**上做；已经配好的机器直接跳到第 4 节启动。

### 2.1 生成专用部署密钥

不用账号级 ssh key，单独给这个仓库一把，泄漏了只影响这一个仓库。

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_ssr -N '' -C "ssr-autosync@$(hostname -s)"
cat ~/.ssh/github_ssr.pub
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh-keygen -t ed25519 -f ~/.ssh/github_ssr -N ''` | 生成无口令的 ed25519 密钥对，专供自动同步 |
| 2 | `cat ~/.ssh/github_ssr.pub` | 打印公钥，复制整行去 GitHub 粘贴 |

然后在 GitHub：仓库 → **Settings → Deploy keys → Add deploy key** → 粘贴公钥 → **勾选 Allow write access**（不勾只能读，push 会失败）。

### 2.2 写 ssh config，让它自动用这把钥匙

```bash
cat >> ~/.ssh/config <<'EOF'

Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_ssr
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cat >> ~/.ssh/config … EOF` | 连 github.com 时固定使用 `github_ssr` 这把密钥；`IdentitiesOnly yes` 防止拿别的 key 去试导致认证被拒 |
| 2 | `chmod 600 ~/.ssh/config` | ssh 会忽略权限过宽的配置文件 |
| 3 | `ssh-keyscan … >> known_hosts` | 预置 github.com 主机指纹，避免首次连接要求确认 |

### 2.3 验证密钥与仓库可写

```bash
ssh -T git@github.com
GIT_SSH_COMMAND="ssh -i ~/.ssh/github_ssr -o IdentitiesOnly=yes -o BatchMode=yes" \
  git ls-remote git@github.com:2200170080/SSR.git | head -3
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh -T git@github.com` | 期望 `Hi 2200170080/SSR! You've successfully authenticated…`（Deploy key 只会显示仓库名，不会显示用户名，这是正常的） |
| 2 | `git ls-remote …` | 能列出 `refs/heads/main` 就说明读权限通；空仓库会什么都不输出但不报错，也算通过 |

### 2.4 在运行根里配置 remote 与提交身份

`origin` 保留官方仓库（对照用），新增 `backup` 指向自己的仓库——所有推送都走 `backup`。

```bash
cd ~/SSR
git remote add backup git@github.com:2200170080/SSR.git
git remote -v
git config user.name "2200170080"
git config user.email "3029316776@qq.com"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd ~/SSR` | 所有 git 操作都在运行根里 |
| 2 | `git remote add backup …` | 新增远端 `backup`；`origin` 保持官方地址 |
| 3 | `git remote -v` | 确认 `backup` 的 fetch/push 都是自己的仓库 |
| 4 | `git config user.name/email` | 设置本仓库的提交身份（自动同步脚本也会用 `-c` 覆盖这两项） |

### 2.5 忽略规则：只挡大二进制

两处要一致：仓库内 `.gitignore`（会同步给别人）+ 本机 `.git/info/exclude`（兜底）。规则只有 6 条——

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
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cat >> .git/info/exclude … EOF` | 登记 6 类不进仓库的内容；前导 `/` 锚定仓库根，否则会误伤 `src/` 下的同名路径 |

`.gitignore` 里对应同一套规则，外加两条**否定规则**——Python 模板默认有全局 `*.log`，不解除的话日志永远同步不上去：

```gitignore
# ---- SSR local run data ----
/models/
/artifacts/multitq_text_index.pkl
/artifacts/multitq_graph_retriever.pkl
/artifacts/llm_cache.jsonl
/MultiTQ/
/.hf_cache/

!/logs/
!/logs/*.log
```

`datasets/` 在 `.gitignore` 里也被忽略，但运行时提示词要跟踪，所以强制加入：

```bash
git -C ~/SSR add -f datasets/MultiTQ/prompt/gp_prompt.txt datasets/MultiTQ/prompt/qa_prompt.txt
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git add -f …/gp_prompt.txt …/qa_prompt.txt` | `-f` 无视 `/datasets/` 规则，把两份提示词纳入版本管理（它们是运行时真正生效的文件） |

### 2.6 首次推送

```bash
cd ~/SSR
git add -A
git status --porcelain | head
git commit -m "chore: initial sync of the run root"
git push -u backup main
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git add -A` | 把全部（未被忽略的）文件加入暂存区 |
| 2 | `git status --porcelain` | 检查有没有把不该进的文件带进来（出现 `models/`、`*.pkl` 就是忽略规则没生效） |
| 3 | `git commit -m …` | 生成第一个同步提交 |
| 4 | `git push -u backup main` | 推送到自己的仓库，并把 `backup/main` 设为上游 |

---

## 3. 自动同步脚本 `tools/autosync.sh`

### 3.1 关键配置（脚本头部，全部可用环境变量覆盖）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `ROOT` | `$HOME/SSR` | 要同步的仓库根 |
| `REMOTE` | `backup` | 推送目标远端名 |
| `BRANCH` | `main` | 推送分支 |
| `INTERVAL` | `60` | 轮询间隔（秒） |
| `LOG` | 由启动命令的 `tee` 决定 | 日志落点 |
| `AUTHOR_NAME` | `2200170080` | 自动提交的作者名 |
| `AUTHOR_EMAIL` | `3029316776@qq.com` | 自动提交的邮箱 |

### 3.2 主循环在做什么

| 步骤 | 动作 | 说明 |
|---|---|---|
| 1 | `git add -A` | 每轮重新暂存，新增/删除/改名都会跟上 |
| 2 | `git diff --cached --quiet` | 暂存区没变化就跳过，不产生空提交 |
| 3 | `git -c user.name=… commit -m "auto: <时间> @ <主机名>"` | 有变化就提交；提交信息带时间与机器名，便于回溯是哪台机改的 |
| 4 | `git push -q "$REMOTE" HEAD:$BRANCH` | 推送；失败（断网/密钥问题）只打印一行，`NEED_PUSH=1` 保留，下一轮重试 |
| 5 | `sleep $INTERVAL` | 每 60 秒一轮 |

设计上有两点值得记住：

- **推送失败不丢数据**：提交已经在本地，下轮继续推；
- **不碰工作区**：只做 add/commit/push，不做 reset/checkout，不会覆盖你正在改的文件。

---

## 4. 启动、查看、停止、重启

### 4.1 启动

```bash
tmux new -d -s sync -c ~/SSR 'bash ~/SSR/tools/autosync.sh 2>&1 | tee -a ~/SSR/logs/autosync.log'
tmux ls
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s sync -c ~/SSR '…'` | 在后台会话 `sync` 里启动脚本，工作目录固定 `~/SSR`；输出同时进屏幕和 `logs/autosync.log` |
| 2 | `tmux ls` | 应看到 `sync: 1 windows (created …)` |

> `logs/autosync.log` 本身也被跟踪，所以「同步器自己的工作日志」也会进仓库——这正是想要的效果：看得见它每轮做了什么。

### 4.2 查看 / 进入 / 退出

```bash
tmux ls                              # 有没有 sync
tmux attach -t sync                  # 进去看实时输出，Ctrl+B 然后 D 脱离（不影响运行）
tail -5 ~/SSR/logs/autosync.log      # 不进会话也能看最近几条
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux ls` | 会话列表，`sync` 在即正常 |
| 2 | `tmux attach -t sync` | 进入会话；`Ctrl+B` `D` 脱离后脚本继续跑 |
| 3 | `tail -5 …/autosync.log` | 看最近几轮是 `commit`/`pushed` 还是 `push failed` |

### 4.3 停止

```bash
tmux kill-session -t sync
pgrep -af autosync.sh || echo "stopped"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux kill-session -t sync` | 结束会话，`while` 循环随之终止 |
| 2 | `pgrep -af autosync.sh` | 确认没有残留进程 |

### 4.4 重启（机器重启后必须执行一次）

这台机器没有 cron，也没有 systemd user session，重启后 `sync` 会话不会自己回来。开机后第一件事：

```bash
tmux new -d -s sync -c ~/SSR 'bash ~/SSR/tools/autosync.sh 2>&1 | tee -a ~/SSR/logs/autosync.log'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s sync -c ~/SSR '…'` | 重新拉起常驻会话；日志追加到同一个 `autosync.log`，不覆盖之前的记录 |

如果希望它跟着实验一起起来，可以写进 `~/SSR/tools/` 的一个小包装脚本，但**别用 `@reboot` cron**——这个账号在共享节点上没有 crontab 权限。

---

## 5. 验证闭环（改一次，确认真的上去了）

```bash
date '+%F %T'
echo "sync check $(date +%s)" >> ~/SSR/README.md
sleep 75
git -C ~/SSR log -1 --format='%h %an <%ae> %s'
GIT_SSH_COMMAND="ssh -i ~/.ssh/github_ssr -o IdentitiesOnly=yes -o BatchMode=yes" \
  git -C ~/SSR ls-remote backup main
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `echo … >> README.md` | 制造一个可提交的改动 |
| 2 | `sleep 75` | 等过一个 60 秒轮询周期 |
| 3 | `git log -1 …` | 应看到新的 `auto: <时间> @ <主机名>`，作者为 `2200170080 <3029316776@qq.com>` |
| 4 | `git ls-remote backup main` | 该 sha 与 `git rev-parse HEAD` 相同即推送成功 |

判据一句话：**本地 HEAD 的 sha == `git ls-remote backup main` 的 sha**。

---

## 6. 日常操作

| 场景 | 命令 |
|---|---|
| 改完想立刻同步（不等 60 秒） | `cd ~/SSR && git add -A && git commit -m "描述" && git push backup main` |
| 只想看远端到了哪 | `git ls-remote backup main` |
| 远端有别人的提交（比如你在网页上改过） | `git pull --rebase backup main` |
| 看某个文件为什么没同步 | `git check-ignore -v 路径` |
| 看自动同步最近做了什么 | `tail -20 ~/SSR/logs/autosync.log` |
| 临时停掉再改历史（如改署名） | `tmux kill-session -t sync` → 改 → 重启 |

---

## 7. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `! [rejected] main -> main (fetch first)` | 远端有你本地没有的提交（网页改过/另一台机器推过） | `git pull --rebase backup main` 再让脚本下轮推 |
| `Permission denied (publickey)` | Deploy key 没配、没勾 write access，或 ssh config 指错密钥 | `ssh -T git@github.com` 复测；确认 `IdentityFile ~/.ssh/github_ssr` |
| 改了文件但一直不提交 | 命中了忽略规则 | `git check-ignore -v <文件>` 看是哪条规则 |
| 日志文件没进仓库 | `.gitignore` 里 Python 模板的全局 `*.log` | 末尾补 `!/logs/` 与 `!/logs/*.log` |
| push 被拒：文件超过 100 MB | GitHub 单文件上限 | 把该文件写进忽略规则，别提交 |
| `logs/` 里的大日志越推越慢 | 日志本身在长大 | 定期归档或把超大日志加入忽略 |
| 机器重启后不同步了 | 没有 cron/systemd，tmux 会话没了 | 按 4.4 手动起一次 |
| 提交作者不对 | 脚本里 `AUTHOR_NAME/AUTHOR_EMAIL` 或仓库 `user.name/email` 没设 | 改脚本头部两行；已经提交的历史用 `git filter-branch` 重写后强推（仅限自己的仓库） |
| 改脚本后服务器执行报 `#!/bin/bash: No such file or directory` | 用了带 BOM 的工具写文件 | 用 UTF-8 **无 BOM** 写回（`UTF8Encoding($false)` / `write` 工具），`head -c 3 \| od -c` 确认是 `# ! /` |

---

## 8. 当前基线（2026-11-02 23:50 实测）

| 项 | 值 |
|---|---|
| 主机 | `A6000x1-3-168`（`ecs-instance-e1330fb1`） |
| 运行根 / 仓库根 | `~/SSR`（`/root/user/cs_tcci_zengdexin/SSR`） |
| 分支 / 远端 | `main` → `backup` = `git@github.com:2200170080/SSR.git`（`origin` 仍指官方 `zhangli-coding/SSR`） |
| 部署密钥 | `~/.ssh/github_ssr`（ed25519，432 B） |
| 提交身份 | `2200170080 <3029316776@qq.com>` |
| 轮询 | 60 秒；tmux 会话 `sync`（2026-11-02 17:43:22 起） |
| 自动提交格式 | `auto: 2026-11-02 23:45:49 @ ecs-instance-e1330fb1` |
| 远端 tip | `2c2bf06` |
| 跟踪文件数 | 22（`src/` 9、`datasets/` 6、`tools/` 2、`figure/` 1、`logs/` 1、`LICENSE`、`README.md`、`.gitignore`） |
| 仓库 `.git` 体积 | 91 MB（历史里曾有 41 MB 的 `.npy` 与全量日志，已从工作树移除但留在历史中） |
| 忽略规则 | `.git/info/exclude` 与 `.gitignore` 各 6 条：`/models/`、两个索引 `*.pkl`、`/artifacts/llm_cache.jsonl`、`/MultiTQ/`、`/.hf_cache/` |
| 自动同步日志 | `~/SSR/logs/autosync.log` |

---

## 9. 与部署手册的关系

- **部署手册**负责「从零把环境跑起来」：拉代码（第 3 节，克隆的就是 `2200170080/SSR`，所以 `tools/autosync.sh` 与 `tools/run_ssr.sh` 随仓库下来）、数据、模型、vLLM、跑实验（第 10 节）。
- **本文**负责「跑起来之后代码怎么持续回到 GitHub」：密钥、remote、忽略规则、脚本、tmux、验证与排错。
- 两者共用的两条铁律：① 仓库根即运行根 `~/SSR`，不要多套一层；② 只挡大二进制，其余全同步。
