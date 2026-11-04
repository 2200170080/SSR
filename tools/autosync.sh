#!/usr/bin/env bash
# autosync.sh - poll ~/SSR for tracked changes, commit and push to a mirror.
# Runs anywhere (no cron / systemd needed): start it inside tmux.
#
#   tmux new -d -s sync -c ~/SSR 'bash ~/SSR/tools/autosync.sh'
#
# Env knobs: ROOT REMOTE BRANCH INTERVAL MAX_MB AUTHOR_NAME AUTHOR_EMAIL
set -uo pipefail

ROOT="${ROOT:-$HOME/SSR}"
REMOTE="${REMOTE:-backup}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
MAX_MB="${MAX_MB:-50}"
AUTHOR_NAME="${AUTHOR_NAME:-2200170080}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-3029316776@qq.com}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_ssr -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"

cd "$ROOT" 2>/dev/null || { echo "[autosync] run root not found: $ROOT"; exit 1; }

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "[autosync] remote '$REMOTE' is not configured yet. Add it first:"
  echo "  git remote add $REMOTE git@github.com:<your-account>/<your-repo>.git"
  echo "[autosync] then test: git push $REMOTE HEAD:$BRANCH"
  exit 1
fi

echo "[autosync] root=$ROOT remote=$REMOTE branch=$BRANCH interval=${INTERVAL}s max=${MAX_MB}MB"
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
    echo -e "[autosync] skipped files bigger than ${MAX_MB}MB (add them to .git/info/exclude):"
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
    ERR=$(git push -q "$REMOTE" "HEAD:$BRANCH" 2>&1)
    if [ $? -eq 0 ]; then
      echo "[autosync] pushed $(git rev-parse --short HEAD) -> $REMOTE/$BRANCH"
      NEED_PUSH=0
    else
      echo "[autosync] push failed, retry in ${INTERVAL}s"
      echo "$ERR" | tail -6 | sed 's/^/    /'
    fi
  fi

  sleep "$INTERVAL"
done
