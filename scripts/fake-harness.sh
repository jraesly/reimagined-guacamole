#!/usr/bin/env bash
# Drives three chat-completions turns through the probe proxy the way a
# coding harness would: a system prompt with a timestamp (which changes on
# turn 3), a tool list, streaming, no stream_options (so probe must inject
# include_usage). Usage: fake-harness.sh <proxy-base-url> <model>
set -euo pipefail
BASE="${1:-http://127.0.0.1:8787/v1}"
MODEL="${2:-qwen3.6:35b-32k}"
UA="fake-harness/1.0"
TOOLS='[{"type":"function","function":{"name":"read_file","description":"Read a file from the workspace","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'
SYS1="You are a coding agent working in /repo. Answer briefly. Current time: 22:41:03"
SYS3="You are a coding agent working in /repo. Answer briefly. Current time: 22:41:19"

turn() { # system, messages-json-array-tail
  local sys="$1" tail="$2"
  local body
  body=$(jq -cn --arg m "$MODEL" --arg s "$sys" --argjson tools "$TOOLS" --argjson tail "$tail" \
    '{model:$m, stream:true, max_tokens:48, messages:([{role:"system",content:$s}] + $tail), tools:$tools}')
  curl -sS -N -A "$UA" -H 'Content-Type: application/json' -H 'Accept-Encoding: gzip' \
    --data-binary "$body" "$BASE/chat/completions" | grep -c '^data:' | sed 's/^/  chunks: /'
}

echo "turn 1 (first turn)"
turn "$SYS1" '[{"role":"user","content":"What does the file main.go do? Use read_file if needed."}]'
echo "turn 2 (append-only: expect prefix ok)"
turn "$SYS1" '[{"role":"user","content":"What does the file main.go do? Use read_file if needed."},{"role":"assistant","content":"It defines the CLI entry point."},{"role":"user","content":"Now list its subcommands."}]'
echo "turn 3 (timestamp changed in the system prompt: expect a break at system[0].content)"
turn "$SYS3" '[{"role":"user","content":"What does the file main.go do? Use read_file if needed."},{"role":"assistant","content":"It defines the CLI entry point."},{"role":"user","content":"Now list its subcommands."},{"role":"assistant","content":"fit, capture, header, version."},{"role":"user","content":"Thanks. One more: which flag sets the context size?"}]'
