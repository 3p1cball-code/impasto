#!/usr/bin/env bash
# Deploy to another machine as a systemd user service on :6868.
#   ./deploy/deploy.sh user@host
# NPG_PORT / NPG_DATA on the target come from deploy/npgcoder.service.
set -euo pipefail
HOST=${1:-${NPG_DEPLOY_HOST:-}}
if [ -z "$HOST" ]; then
  echo "usage: ./deploy/deploy.sh user@host   (or set NPG_DEPLOY_HOST)" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
rsync -a --delete --exclude .venv --exclude _x --exclude data --exclude __pycache__ --exclude .pytest_cache \
      --exclude .playwright-mcp --exclude '*.gcode' --exclude '*.3mf' --exclude .git ./ "$HOST:non-planar-gcoder/"
ssh "$HOST" 'set -e; cd ~/non-planar-gcoder
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip >/dev/null
  .venv/bin/pip install -q -r requirements.txt
  mkdir -p ~/.config/systemd/user data
  cp deploy/npgcoder.service ~/.config/systemd/user/
  systemctl --user daemon-reload
  systemctl --user enable --now npgcoder.service
  systemctl --user restart npgcoder.service
  sleep 2; systemctl --user is-active npgcoder.service; curl -s localhost:6868/api/version; echo'
