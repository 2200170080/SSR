#!/usr/bin/env bash
# autosync.sh - poll ~/SSR for tracked changes, commit and push to a mirror.
# Runs anywhere (no cron / systemd needed): start it inside tmux.
#
#   tmux new -d -s sync -c ~/SSR 'bash ~/SSR/tools/autosync.sh'
#
# Env knobs: ROOT REMOTE BRANCH INTERVAL MAX_MB RETRIES AUTHOR_NAME AUTHOR_EMAIL
set -uo pipefail

ROOT="${ROOT:-$HOME/SSR}"
REMOTE="${REMOTE:-backup}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
MAX_MB="${MAX_MB:-50}"
RETRIES="${RETRIES:-3}"
AUTHOR_NAME="${AUTHOR_NAME:-2200170080}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-3029316776@qq.com}"

cd "$ROOT" 2>/dev/null || { echo "[autosync] run root not found: $ROOT"; exit 1; }

# --- ssh key: keep a local-disk copy (home is on CephFS) + connection reuse ---
SRC_KEY="$HOME/.ssh/github_ssr"
LOCAL_KEY="/tmp/.autosync_gh_key"
CTL_DIR="/tmp/.autosync_ssh"
mkdir -p "$CTL_DIR"
chmod 700 "$CTL_DIR" 2>/dev/null
if [ -r "$SRC_KEY" ]; then
  cp -f "$SRC_KEY" "$LOCAL_KEY" && chmod 600 "$LOCAL_KEY"
elif [ ! -r "$LOCAL_KEY" ]; then
  echo "[autosync] no deploy key at $SRC_KEY, and no local copy"; exit 1
fi
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -i $LOCAL_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
 -o ControlMaster=auto -o ControlPath=$CTL_DIR/gh-%r@%h:%p -o ControlPersist=10m -o ServerAliveInterval=15"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "[autosync] remote '$REMOTE' is not configured yet. Add it first:"
  echo "  git remote add $REMOTE git@github.com:<your-account>/<your-repo>.git"
  echo "[autosync] then test: git push $REMOTE HEAD:$BRANCH"
  exit 1
fi

echo "[autosync] root=$ROOT remote=$REMOTE branch=$BRANCH interval=${INTERVAL}s max=${MAX_MB}MB retries=${RETRIES}"
echo "[autosync] everything except large binaries is tracked; files over ${MAX_MB}MB are skipped"

NEED_PUSH=0
while true; do
  git add -A >/dev/null 2>&1

  # --- guard: never stage a file that GitHub will reject (100 MiB hard limit) ---
  BIG=""
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || continue
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt $((MAX_MB * 1024 * 1024)) ]; then
      BIG="$BIG$f ($((sz / 1024 / 1024))MB)\n"
      git reset -q -- "$f" 2>/dev/null
    fi
  done < <(git diff --cached --name-only)
  if [ -n "$BIG" ]; then
    echo -e "[autosync] skipped files bigger than ${MAX_MB}MB (add them to .gitignore):"
    echo -e "$BIG" | sed '/^$/d; s/^/  /'
  fi

  if ! git diff --cached --quiet; then
    MSG="auto: $(date '+%F %T') @ $(hostname -s)"
    if git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit -q -m "$MSG"; then
      echo "[autosync] commit $MSG"
      NEED_PUSH=1
    else
      echo "[autosync] commit failed, will retry"
    fi
  fi

  if [ "$NEED_PUSH" = "1" ]; then
    n=1
    while : ; do
      ERR=$(git push "$REMOTE" "HEAD:$BRANCH" 2>&1)
      if [ $? -eq 0 ]; then
        echo "[autosync] pushed $(git rev-parse --short HEAD) -> $REMOTE/$BRANCH"
        NEED_PUSH=0
        break
      fi
      echo "[autosync] push attempt $n/$RETRIES failed"
      echo "$ERR" | tail -4 | sed 's/^/    /'
      if [ "$n" -ge "$RETRIES" ]; then
        echo "[autosync] giving up this round, retry in ${INTERVAL}s"
        break
      fi
      n=$((n + 1))
      sleep 15
    done
  fi

  sleep "$INTERVAL"
done
