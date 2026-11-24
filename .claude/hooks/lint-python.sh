#!/usr/bin/env bash
# PostToolUse hook: auto-format and lint any Python file Claude just edited.
# Exit code 2 feeds the lint errors back to Claude so it fixes them immediately.
set -euo pipefail

file=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))')

[[ "$file" == *.py ]] || exit 0
[[ -f "$file" ]] || exit 0
command -v uv >/dev/null || exit 0

cd "$CLAUDE_PROJECT_DIR"
uv run --quiet ruff format "$file" >/dev/null 2>&1 || true
if ! out=$(uv run --quiet ruff check --fix "$file" 2>&1); then
  echo "ruff found problems in $file:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
