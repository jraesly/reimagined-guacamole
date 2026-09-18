#!/usr/bin/env bash
# Requires jq; live runs also require curl and ollama. Dry runs never contact Ollama.
set -euo pipefail

dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1; shift ;;
    "") ;;
    *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi
command -v jq >/dev/null || { printf 'jq is required\n' >&2; exit 2; }
if [ "$dry_run" -eq 0 ]; then
    command -v curl >/dev/null
    command -v ollama >/dev/null
fi
repo_root=$(cd "$(dirname "$0")/.." && pwd)
probe=${PROBE:-/tmp/probe}
if [ -z "${PROBE:-}" ]; then
    (
        export PATH=/opt/homebrew/bin:$PATH GOCACHE=/tmp/gocache GOPATH=/tmp/gopath GOFLAGS=-mod=mod
        cd "$repo_root"
        go build -o "$probe" ./cmd/probe
    )
fi
work=$(mktemp -d "${TMPDIR:-/tmp}/probe-validation.XXXXXX")
active_model=''
endpoint=http://127.0.0.1:11434/api/generate
unload() {
    local body
    body=$(jq -nc --arg model "$1" '{model:$model,keep_alive:0}')
    curl --silent --show-error --fail --max-time 60 -H 'Content-Type: application/json' \
        --data "$body" "$endpoint" > "$work/unload.json" &&
        jq -e 'type == "object" and (.error // "") == ""' "$work/unload.json" >/dev/null
}
cleanup() {
    if [ -n "$active_model" ] && [ "$dry_run" -eq 0 ]; then
        unload "$active_model" || printf 'Warning: failed to unload %s\n' "$active_model" >&2
    fi
    rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$probe" fit --json --context 8k,16k,32k,64k,128k,256k > "$work/fit.json" 2> "$work/fit.err"; then
    cat "$work/fit.err" >&2
    # Sandboxed Macs may deny sysctl, and machines may have no local models.
    # Examples are explicitly labeled and never emitted as validation evidence.
    if [ "$dry_run" -eq 1 ]; then
        printf 'DRY RUN: probe fit unavailable; using illustrative requests.\n'
        printf '{"hardware":null,"models":[]}\n' > "$work/fit.json"
    else
        exit 1
    fi
fi
jq -e 'type == "object" and (.models | type == "array")' "$work/fit.json" >/dev/null
# One Ollama alias per model avoids duplicate work for shared blobs.
jq -r '
    .models[] | select((.error // "") == "") |
    ([.name] + (.aliases // []) | map(select(type == "string" and contains(":"))) | unique | .[0]) as $name |
    select($name != null) |
    (.rows // []) as $rows |
    [($rows | map(select(.Verdict == "yes" or .Verdict == "tight")) | sort_by(.Context) | last),
     ($rows | map(select(.Verdict == "no")) | sort_by(.Context) | first)][] |
    select(. != null) | [$name, .Context, .Verdict] | @tsv
' "$work/fit.json" > "$work/targets.tsv"
if [ ! -s "$work/targets.tsv" ]; then
    if [ "$dry_run" -eq 0 ]; then
        printf 'No Ollama aliases with fit verdicts found; nothing validated.\n' >&2
        exit 1
    fi
    printf 'DRY RUN: no usable Ollama targets; using illustrative requests (not measurements).\n'
    printf 'example:latest\t8192\tyes\nexample:latest\t16384\tno\n' > "$work/targets.tsv"
fi

printf '%-34s %8s %-9s %-20s %9s %s\n' MODEL CTX PREDICTED PROCESSOR TOK/S RESULT
: > "$work/results.jsonl"
failed=0
while IFS="$(printf '\t')" read -r name ctx predicted; do
    body=$(jq -nc --arg model "$name" --argjson ctx "$ctx" \
        '{model:$model,prompt:"Say hi",stream:false,keep_alive:"1m",options:{num_ctx:$ctx,num_predict:8}}')
    if [ "$dry_run" -eq 1 ]; then
        printf 'DRY RUN POST %s %s\n' "$endpoint" "$body"
        printf 'DRY RUN ollama ps: inspect PROCESSOR for %s (predicted %s)\n' "$name" "$predicted"
        printf 'DRY RUN POST %s %s\n' "$endpoint" "$(jq -nc --arg model "$name" '{model:$model,keep_alive:0}')"
        continue
    fi
    active_model=$name
    processor=unknown
    rate=null
    result=FAIL
    detail=''
    if status=$(curl --silent --show-error --max-time 600 -o "$work/response.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' --data "$body" "$endpoint"); then
        if jq -e 'type == "object"' "$work/response.json" >/dev/null 2>&1; then
            error=$(jq -r '.error // empty' "$work/response.json")
            if [ -n "$error" ]; then
                processor='Ollama error'
                detail=$error
                if [ "$predicted" = no ]; then result=PASS; fi
            elif [ "$status" != 200 ]; then
                detail="HTTP $status without an Ollama error"
            elif [ "$(jq -r '.done // false' "$work/response.json")" != true ]; then
                detail='Ollama response did not complete generation'
            elif ollama ps > "$work/ps.txt"; then
                processor=$(awk -v model="$name" '
                    NR == 1 {
                        start = index($0, "PROCESSOR")
                        finish = index($0, "CONTEXT")
                        if (!finish) finish = index($0, "UNTIL")
                        next
                    }
                    $1 == model && start > 0 {
                        s = finish > start ? substr($0, start, finish-start) : substr($0, start)
                        gsub(/^[ \t]+|[ \t]+$/, "", s)
                        print s; exit
                    }
                ' "$work/ps.txt")
                processor=${processor:-unknown}
                rate=$(jq -c 'if (.eval_duration | type) == "number" and .eval_duration > 0 and
                    (.eval_count | type) == "number" then .eval_count * 1000000000 / .eval_duration else null end' "$work/response.json")
                if [ "$processor" = '100% GPU' ]; then
                    if [ "$predicted" != no ]; then
                        result=PASS
                    else
                        detail='fit was too pessimistic: predicted no, observed 100% GPU'
                    fi
                elif [[ "$processor" == *CPU* ]]; then
                    if [ "$predicted" = no ]; then result=PASS; else detail='predicted fit, observed CPU offload'; fi
                else
                    detail='could not establish processor allocation for this model'
                fi
            else
                detail='ollama ps failed'
            fi
        else
            detail='invalid Ollama JSON response'
        fi
    else
        detail='request transport failed; no fit conclusion possible'
    fi
    unload_failed=0
    if unload "$name"; then
        active_model=''
    else
        result=FAIL
        detail="$detail; unload failed, stopping to avoid contaminating subsequent runs"
        unload_failed=1
    fi
    printf '%-34s %8s %-9s %-20s %9s %s\n' "$name" "$ctx" "$predicted" "$processor" "$rate" "$result"
    if [ -n "$detail" ]; then printf '  %s\n' "$detail"; fi
    jq -nc --arg model "$name" --argjson ctx "$ctx" --arg predicted "$predicted" \
        --arg processor "$processor" --argjson tok_s "$rate" --arg result "$result" --arg detail "$detail" \
        '{model:$model,ctx:$ctx,predicted:$predicted,observed_processor:$processor,tok_s:$tok_s,result:$result,detail:$detail}' >> "$work/results.jsonl"
    if [ "$result" = FAIL ]; then failed=1; fi
    if [ "$unload_failed" -eq 1 ]; then break; fi
done < "$work/targets.tsv"

if [ "$dry_run" -eq 1 ]; then
    printf 'DRY RUN: no Ollama calls made; no validation report written.\n'
    exit 0
fi
uname -a > "$work/uname.txt"
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$work/gpu.txt" 2>&1; then
    gpu_tool=nvidia-smi
elif command -v rocm-smi >/dev/null && rocm-smi --showproductname > "$work/gpu.txt" 2>&1; then
    gpu_tool=rocm-smi
else
    gpu_tool=unavailable
    printf 'GPU diagnostic command unavailable or failed\n' > "$work/gpu.txt"
fi
host=$(hostname | tr -c 'A-Za-z0-9._-' '_' | sed 's/_$//')
output="docs/validation/${host}-$(date +%Y%m%d).json"
mkdir -p docs/validation
jq -n --slurpfile fit "$work/fit.json" --slurpfile results "$work/results.jsonl" \
    --rawfile uname "$work/uname.txt" --rawfile gpu "$work/gpu.txt" --arg gpu_tool "$gpu_tool" \
    '{hardware:$fit[0].hardware,uname:$uname,gpu_tool:$gpu_tool,gpu_output:$gpu,results:$results}' > "$output"
printf 'Wrote %s\n' "$output"
exit "$failed"
