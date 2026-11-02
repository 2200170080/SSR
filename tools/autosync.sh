#!/usr/bin/env bash
# autosync.sh - poll ~/SSR for tracked code changes, commit and push to a mirror.
# Runs anywhere (no cron / systemd needed): start it inside tmux.
#
#   tmux new -d -s sync -c ~/SSR 'bash ~/SSR/tools/autosync.sh'
#
# Env knobs: ROOT REMOTE BRANCH INTERVAL LOG AUTHOR_NAME AUTHOR_EMAIL
set -uo pipefail

ROOT="${ROOT:-$HOME/SSR}"
REMOTE="${REMOTE:-backup}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
AUTHOR_NAME="${AUTHOR_NAME:-Structured Subgraph Retrieval}"
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

echo "[autosync] root=$ROOT remote=$REMOTE branch=$BRANCH interval=${INTERVAL}s"
echo "[autosync] tracked code only; data/models/artifacts/logs are excluded"

NEED_PUSH=0
while true; do
  git add -A >/dev/null 2>&1
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
    if git push -q "$REMOTE" "HEAD:$BRANCH" >/dev/null 2>&1; then
      echo "[autosync] pushed $(git rev-parse --short HEAD) -> $REMOTE/$BRANCH"
      NEED_PUSH=0
    else
      echo "[autosync] push failed (offline or key issue), retry in ${INTERVAL}s"
    fi
  fi

  sleep "$INTERVAL"
done
