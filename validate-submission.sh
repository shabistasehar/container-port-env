#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [[ -z "${PING_URL}" ]]; then
  echo "Usage: ./validate-submission.sh <ping_url> [repo_dir]"
  exit 1
fi

cd "${REPO_DIR}"

echo "== Health =="
curl -fsS "${PING_URL%/}/health"
echo

echo "== Reset =="
curl -fsS -X POST "${PING_URL%/}/reset" \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"easy"}'
echo

echo "== OpenEnv Validate =="
openenv validate --url "${PING_URL%/}"

echo "== Docker Build =="
docker build -t container-port-env .

echo "== Inference =="
python inference.py --url "${PING_URL%/}" --difficulty all

echo "Validation complete."
