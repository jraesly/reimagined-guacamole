# Validating `fit` against a real runtime

`probe fit` predicts whether a model fits at a given context. `scripts/validate-linux.sh`
checks those predictions against what Ollama actually does on the same machine,
so a verdict is backed by an observation, not just arithmetic. The script is
written for a Linux box with an NVIDIA or AMD GPU, but the mechanics are the
same on macOS and it has been run there.

## Prerequisites

- Ollama running locally (`http://127.0.0.1:11434`) with at least one model
  pulled; `ollama ps` must work.
- `jq`, `curl`, `bash` 3.2+ (no associative arrays are used).
- Go 1.22+ if the script has to build probe; otherwise `PROBE=/path/to/probe`.
- Enough free time: each check loads a model once (10–60 s) and generates 8
  tokens; four checks is typical.

## Run

```
scripts/validate-linux.sh              # builds /tmp/probe if PROBE is unset
PROBE=./bin/probe scripts/validate-linux.sh
scripts/validate-linux.sh --dry-run    # prints the requests, contacts nothing
```

For every Ollama model `fit` knows, the script takes two boundary contexts from
`probe fit --json --context 8k,16k,32k,64k,128k,256k`:

- `CTX_OK`: the largest context with verdict `yes` or `tight`
- `CTX_NO`: the smallest context with verdict `no` (if any)

For each it loads the model at exactly that context (`/api/generate` with
`num_ctx`, `keep_alive` 1m, `num_predict` 8), reads the `PROCESSOR` column of
`ollama ps`, records the decode rate from `eval_count / eval_duration`, and
unloads the model.

## Reading the table

```
MODEL          CTX PREDICTED PROCESSOR          TOK/S RESULT
qwen3.8:27b  65536 yes       100% GPU           14.19 PASS
qwen3.8:27b 131072 no        28%/72% CPU/GPU     7.91 PASS
qwen3.6:35b 131072 tight     11%/89% CPU/GPU    31.50 FAIL   predicted fit, observed CPU offload
qwen3.6:35b 262144 no        27%/73% CPU/GPU    10.20 PASS
```

- `PASS` at a predicted `yes`/`tight`: Ollama placed 100% of the model on
  the GPU.
- `PASS` at a predicted `no`: Ollama spilled to CPU (or refused). `fit` was
  right that it would not fit.
- `FAIL … too pessimistic`: `fit` said `no` but the runtime kept everything
  on the GPU. Usually the reserve is too large for this machine or the model
  has a cache the formula overestimates (MLA, sliding window).
- `FAIL … predicted fit, observed CPU offload`: `fit` said `yes`/`tight` but
  the runtime spilled. At `tight` this means the runtime's own buffers took
  the last few hundred MB; at `yes` it is a real modelling error worth an
  issue.

The run above is real (Apple M1 Max, 32 GB, Ollama 0.34, 2026-09-17,
`docs/validation/MacBookPro-20260917.json`). Three of four predictions held;
the `tight` miss is why the compute reserve was raised from 0.5 GB to 1.0 GB and
why every `tight` row now carries a note that it may still spill.

The script exits non-zero when any row is `FAIL`.

## Where results go

`docs/validation/<hostname>-<YYYYMMDD>.json` holds the table plus the
`hardware` block from `probe fit --json`, `uname -a`, and the GPU tool's
output (`nvidia-smi --query-gpu=name,driver_version` or
`rocm-smi --showproductname`). Attach that file to an issue when reporting a
wrong verdict; it contains no prompt text.

## Status by platform

| Platform | Status |
|---|---|
| macOS, Apple Silicon (M1 Max 32 GB) | run 2026-09-17: 3/4 PASS, see above |
| Linux, NVIDIA RTX 4090 24 GB | pending — the script is ready, the box has not been run yet |
| Linux, AMD (ROCm) | detection implemented, never run on hardware |

## Testing the script itself

`scripts/validate_linux_test.py` runs the script against a fake Ollama and a
fake `probe`, covering nine scenarios (all pass, error, pessimistic,
optimistic, transport failure, invalid JSON, `ollama ps` failure, missing
model, unload failure) plus the dry run:

```
python3 scripts/validate_linux_test.py
```
