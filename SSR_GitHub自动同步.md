# SSR 项目自动同步到 GitHub —— 全流程

> 适用环境：`A6000x1-3-168`（`ecs-instance-e1330fb1`），共享盘 `172.30.20.24:6789:/ltaidc`，运行根 `~/SSR`。
> 这套机制让服务器上改的代码、数据、日志、提示词每 60 秒检查一次，**有变化就** commit + push 到 `git@github.com:2200170080/SSR.git`，只有大二进制（模型、检索索引、响应缓存）和同步器自己的日志不进仓库。
> 相关文档：部署手册 `SSR_部署手册.md`（第 3 节拉代码、第 10 节跑实验）；本文只讲**同步**。

---

## 0. 整体是什么

```text
服务器 ~/SSR（git 仓库，仓库根 = 运行根）
        │
        │  tmux 会话 sync 里常驻 scripts/github-sync/autosync.sh，每 60 s：
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

两处要一致：仓库内 `.gitignore`（会同步给别人）+ 本机 `.git/info/exclude`（兜底）。规则共 7 条——

```bash
cat >> ~/SSR/.git/info/exclude <<'EOF'

# never commit: model weights, built indexes, response cache, upstream dataset repo
/models/
/artifacts/*.pkl
/artifacts/llm_cache.jsonl
/artifacts/*/llm_cache.jsonl
/MultiTQ/
/.hf_cache/
/scripts/github-sync/autosync.log
EOF
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cat >> .git/info/exclude … EOF` | 登记 7 类不进仓库的内容；前导 `/` 锚定仓库根，否则会误伤 `src/` 下的同名路径；`artifacts/` 只排掉索引 `*.pkl` 与响应缓存 `llm_cache.jsonl`，**其余实验结果照常同步**；`/scripts/github-sync/autosync.log` 必须挡——它每轮都在写，跟踪它会让同步器自己制造无穷提交 |

`.gitignore` 里对应同一套规则，另外还要加**否定规则**——Python 模板默认有全局 `*.log`，不解除的话 `logs/` 与 `artifacts/*/A.log` 永远同步不上去；否定规则**必须写在 `.gitignore` 里**（`.git/info/exclude` 的 `!` 压不过 `.gitignore`）：

```text
# ---- SSR local run data ----
/models/
/artifacts/*.pkl
/artifacts/llm_cache.jsonl
/artifacts/*/llm_cache.jsonl
/MultiTQ/
/.hf_cache/

# 放行日志：Python 模板有全局 *.log，不解除则 logs/ 与 artifacts/*/A.log 永远同步不上
!/logs/
!/logs/*.log
!/artifacts/**/*.log

# 同步器自己的工作日志：每轮都写，跟踪它 = 每轮一个新提交
/scripts/github-sync/autosync.log
```

判断某个文件到底进不进仓库，用 `git check-ignore -v <路径>` 看命中的是哪一条：

```bash
git -C ~/SSR check-ignore -v artifacts/gp-holdout-2eEr22/A.jsonl   # 期望：无输出（要同步）
git -C ~/SSR check-ignore -v artifacts/gp-B-full-MqkyPV/llm_cache.jsonl  # 期望：命中 /artifacts/*/llm_cache.jsonl
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `git check-ignore -v artifacts/.../A.jsonl` | 无输出 = 会同步；打印规则行 = 被挡，需改规则 |
| 2 | `git check-ignore -v artifacts/.../llm_cache.jsonl` | 应命中 `llm_cache.jsonl` 规则，证明大缓存确实挡在门外 |


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

## 3. 自动同步脚本 `scripts/github-sync/autosync.sh`

### 3.1 关键配置（脚本头部，全部可用环境变量覆盖）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `ROOT` | `$HOME/SSR` | 要同步的仓库根 |
| `REMOTE` | `backup` | 推送目标远端名 |
| `BRANCH` | `main` | 推送分支 |
| `INTERVAL` | `60` | 轮询间隔（秒） |
| `MAX_MB` | `50` | 单文件软阈值：超过就本轮跳过并打印，绝不让 GitHub 因 100 MB 上限拒收 |
| `RETRIES` | `3` | 一轮内 push 失败的重试次数（间隔 15 秒），三次都失败才等下轮 |
| `AUTHOR_NAME` | `2200170080` | 自动提交的作者名 |
| `AUTHOR_EMAIL` | `3029316776@qq.com` | 自动提交的邮箱 |

### 3.2 主循环在做什么

| 步骤 | 动作 | 说明 |
|---|---|---|
| 1 | `git add -A` | 每轮重新暂存，新增/删除/改名都会跟上 |
| 2 | 超大文件守卫 | 遍历暂存区，超过 `MAX_MB` 的 `git reset` 撤下并打印 `skipped files bigger than 50MB`——挡在提交之前，不会被 GitHub 拒 |
| 3 | `git diff --cached --quiet` | 暂存区没变化就跳过，**不产生空提交**（所以「没有新提交」通常意味着服务器真的没有文件变化） |
| 4 | `git -c user.name=… commit -m "auto: <时间> @ <主机名>"` | 有变化就提交；提交信息带时间与机器名，便于回溯是哪台机改的 |
| 5 | `git push "$REMOTE" HEAD:$BRANCH` | 推送；失败打印 `push attempt n/3 failed` + stderr 末 4 行，轮内重试 3 次，仍失败则 `NEED_PUSH=1` 留给下一轮 |
| 6 | `sleep $INTERVAL` | 每 60 秒一轮 |

设计上有三点值得记住：

- **推送失败不丢数据**：提交已经在本地，下轮继续推；
- **不碰工作区**：只做 add/commit/push，不做 reset/checkout，不会覆盖你正在改的文件；
- **只认变化**：没有文件变化就没有提交，日志里也会安静——这不等于同步坏了。

### 3.3 密钥为什么要复制到本地盘、为什么要复用连接

两件事都是踩坑后加的：

1. **密钥复制到 `/tmp/.autosync_gh_key`**：家目录 `~` 在 CephFS（`172.30.20.24:6789:/ltaidc`）上，共享盘读小文件偶发抖动；脚本启动时把 `~/.ssh/github_ssr` 复制一份到本地盘（600 权限），ssh 只用这份。原密钥仍保留，重装/换机不受影响。
2. **`ControlMaster` 复用连接**：每分钟新建一条 ssh 连接去 GitHub，会触发**认证限流**——表现是随机 `Permission denied (publickey)`，而手动 `ssh -T` 又是通的。脚本用 `ControlMaster=auto` + `ControlPath=/tmp/.autosync_ssh/gh-%r@%h:%p` + `ControlPersist=10m`，10 分钟内复用同一条连接，认证次数从「每分钟 1 次」降到「十分钟 1 次」。

脚本里对应的两段（`scripts/github-sync/autosync.sh` 第 21–34 行）：复制密钥 → 导出 `GIT_SSH_COMMAND`；再加 `ServerAliveInterval=15` 防止连接被中间设备静默断开。

---

## 4. 启动、查看、停止、重启

### 4.1 启动

```bash
tmux new -d -s sync -c ~/SSR 'bash ~/SSR/scripts/github-sync/autosync.sh 2>&1 | tee -a ~/SSR/scripts/github-sync/autosync.log'
tmux ls
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s sync -c ~/SSR '…'` | 在后台会话 `sync` 里启动脚本，工作目录固定 `~/SSR`；输出同时进屏幕和 `scripts/github-sync/autosync.log` |
| 2 | `tmux ls` | 应看到 `sync: 1 windows (created …)` |

> `scripts/github-sync/autosync.log` **不进仓库**（已加入忽略规则）：它每轮都在写，跟踪它会让同步器自己制造无穷提交——这是踩过的坑，见 7 节「每分钟都冒出 `auto:` 提交」。

### 4.2 查看 / 进入 / 退出

```bash
tmux ls                              # 有没有 sync
tmux attach -t sync                  # 进去看实时输出，Ctrl+B 然后 D 脱离（不影响运行）
tail -5 ~/SSR/scripts/github-sync/autosync.log      # 不进会话也能看最近几条
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
tmux new -d -s sync -c ~/SSR 'bash ~/SSR/scripts/github-sync/autosync.sh 2>&1 | tee -a ~/SSR/scripts/github-sync/autosync.log'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new -d -s sync -c ~/SSR '…'` | 重新拉起常驻会话；日志追加到同一个 `autosync.log`，不覆盖之前的记录 |

如果希望它跟着实验一起起来，可以写进 `~/SSR/scripts/` 的一个小包装脚本，但**别用 `@reboot` cron**——这个账号在共享节点上没有 crontab 权限。

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

反过来，如果两边 sha 一致但你在网页上看不到“新东西”，先确认服务器上**有没有文件真的变了**：

```bash
cd ~/SSR && find . -path ./.git -prune -o -type f -newermt '-10 minutes' -print | grep -v '^./models' | head
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `find … -newermt '-10 minutes'` | 列出最近 10 分钟内被写过的文件；**空输出 = 没有内容可同步**，此时没有新提交是正常的 |
| 2 | `head` | 只看前几条，避免日志目录刷屏 |

---

## 6. 日常操作

| 场景 | 命令 |
|---|---|
| 改完想立刻同步（不等 60 秒） | `cd ~/SSR && git add -A && git commit -m "描述" && git push backup main` |
| 只想看远端到了哪 | `git ls-remote backup main` |
| 远端有别人的提交（比如你在网页上改过） | `git pull --rebase backup main` |
| 看某个文件为什么没同步 | `git check-ignore -v 路径` |
| 看自动同步最近做了什么 | `tail -20 ~/SSR/scripts/github-sync/autosync.log` |
| 临时停掉再改历史（如改署名） | `tmux kill-session -t sync` → 改 → 重启 |

---

## 7. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `! [rejected] main -> main (fetch first)` | 远端有你本地没有的提交（网页改过/另一台机器推过） | `git pull --rebase backup main` 再让脚本下轮推 |
| `Permission denied (publickey)` | Deploy key 没配、没勾 write access，或 ssh config 指错密钥 | `ssh -T git@github.com` 复测；确认 `IdentityFile ~/.ssh/github_ssr` |
| 改了文件但一直不提交 | 命中了忽略规则 | `git check-ignore -v <文件>` 看是哪条规则 |
| 日志文件没进仓库 | `.gitignore` 里 Python 模板的全局 `*.log` | 末尾补 `!/logs/`、`!/logs/*.log`、`!/artifacts/**/*.log`（必须写在 `.gitignore` 里） |
| push 被拒：文件超过 100 MB | GitHub 单文件硬上限 | 写进忽略规则；脚本另有 50 MB 软阈值，会自动跳过并打印 `skipped files bigger than 50MB` |
| 推送一直失败、本地领先几百个提交 | 某个大文件已经进了历史（哪怕后来删了，blob 还在待推的提交里） | 别硬推：`git branch save-<date> HEAD` → `git reset --soft backup/main` → 重新 `git add -A` 提交一次 → push |
| `logs/` 里的大日志越推越慢 | 日志本身在长大 | 定期归档或把超大日志加入忽略 |
| 机器重启后不同步了 | 没有 cron/systemd，tmux 会话没了 | 按 4.4 手动起一次 |
| 提交作者不对 | 脚本里 `AUTHOR_NAME/AUTHOR_EMAIL` 或仓库 `user.name/email` 没设 | 改脚本头部两行；已经提交的历史用 `git filter-branch` 重写后强推（仅限自己的仓库） |
| 改脚本后服务器执行报 `#!/bin/bash: No such file or directory` | 用了带 BOM 的工具写文件 | 用 UTF-8 **无 BOM** 写回（`UTF8Encoding($false)` / `write` 工具），`head -c 3 \| od -c` 确认是 `# ! /` |
| 每分钟都冒出一条 `auto:` 提交、日志越滚越大 | `scripts/github-sync/autosync.log` 曾被跟踪，同步器每轮写日志 → 每轮新提交 → 每轮新推送 | 已加入忽略并从索引移除：`git rm --cached scripts/github-sync/autosync.log` + 规则 `/scripts/github-sync/autosync.log`；自激循环就此断掉 |
| 每隔几轮 `Permission denied (publickey)`，手动 `ssh -T` 却是通的 | 每分钟新建 ssh 连接，触发 GitHub 认证限流（突发测试连发 10 次全被拒，隔一会儿单次成功） | 已改连接复用：`ControlMaster=auto` + `ControlPath=/tmp/.autosync_ssh/…` + `ControlPersist=10m`，并把密钥复制到本地盘 `/tmp/.autosync_gh_key` |
| `remote: Internal Server Error` / `[remote rejected] … (Internal Server Error)` | GitHub 侧瞬时故障，不是你的配置问题 | 脚本轮内重试 3 次（间隔 15 秒）；连续出现只需等，别改配置 |
| 服务器在跑实验，但 GitHub 上没有新提交 | 跑批已停或没写文件——**没有变化就没有提交** | 先 `ps` 看进程、再看目标日志 mtime，最后才怀疑同步；用第 5 节那条 `find -newermt` 一秒判定 |

---

## 8. 当前基线（2026-11-04 20:35 实测）

| 项 | 值 |
|---|---|
| 主机 | `A6000x1-3-168`（`ecs-instance-e1330fb1`） |
| 运行根 / 仓库根 | `~/SSR`（`/root/user/cs_tcci_zengdexin/SSR`） |
| 分支 / 远端 | `main` → `backup` = `git@github.com:2200170080/SSR.git`（`origin` 仍指官方 `zhangli-coding/SSR`） |
| 部署密钥 | `~/.ssh/github_ssr`（ed25519，432 B）；运行时副本 `/tmp/.autosync_gh_key`（600） |
| 提交身份 | `2200170080 <3029316776@qq.com>` |
| 轮询 | 60 秒；tmux 会话 `sync`（2026-11-04 19:25 起） |
| 自动提交格式 | `auto: 2026-11-04 20:35:09 @ ecs-instance-e1330fb1` |
| 远端 tip | `f852e71`（与本地 HEAD 相同） |
| 跟踪文件数 | **93**（`artifacts/` 66、`src/` 9、`datasets/` 6、`logs/` 3、`tools/` 2、`figure/` 1、根目录 6） |
| `artifacts/` 被跟踪的量 | 66 个文件、共 **22 MB**（`*.txt`、`*.jsonl`、`*.json`、`*.log`） |
| 仓库 `.git` 体积 | **997 MB**（历史里仍有 740 MB / 104 MB 的 `llm_cache.jsonl` 与 41 MB `.npy` blob） |
| 忽略规则 | 7 条：`/models/`、`/artifacts/*.pkl`、`/artifacts/llm_cache.jsonl`、`/artifacts/*/llm_cache.jsonl`、`/MultiTQ/`、`/.hf_cache/`、`/scripts/github-sync/autosync.log`；`.gitignore` 另有三条否定规则放行 `logs/` 与 `artifacts/**/*.log` |
| 脚本 | `~/SSR/scripts/github-sync/autosync.sh`，`INTERVAL=60 MAX_MB=50 RETRIES=3` |
| 自动同步日志 | `~/SSR/scripts/github-sync/autosync.log`（**不跟踪**，仅本地查看） |

---

## 9. 与部署手册的关系

- **部署手册**负责「从零把环境跑起来」：拉代码（第 3 节，克隆的就是 `2200170080/SSR`，所以 `scripts/github-sync/autosync.sh` 随仓库下来）、数据、模型、vLLM、跑实验（第 10 节）。
- **本文**负责「跑起来之后代码怎么持续回到 GitHub」：密钥、remote、忽略规则、脚本、tmux、验证与排错。
- 两者共用的两条铁律：① 仓库根即运行根 `~/SSR`，不要多套一层；② 只挡大二进制，其余全同步。
