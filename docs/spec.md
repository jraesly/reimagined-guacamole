# v0.1 Spec — `fit` + harness capture proxy

Working name: `probe` (rename later). One binary, two commands. `fit` is built first.

## Promise
1. `probe fit`: given this machine and a model file, say whether it fits, at what context, and roughly how fast — from arithmetic, not opinion — and flag when what you downloaded is a fine-tune rather than a baseline.
2. `probe capture`: point your coding harness at `probe` instead of your local server and, per turn, see how many tokens the harness added, how many were re-prefilled because the prompt prefix changed, and what share of the turn's latency was prefill.

## Non-goals (v0.1)
No model rankings or benchmark scores, no fine-tune catalog, no downloads, no server lifecycle, no MCP setup, no daemon, no history DB, no leaderboards, no HTML, no ROCm, no Ollama native API, no automatic config edits.

## User flow
```
probe fit                                       # scans LM Studio / Ollama / llama.cpp model dirs on this machine
probe fit ~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/*.gguf --context 32k
probe fit --suggest                             # <=5 baseline models for this memory class, dated list

probe capture --upstream http://localhost:1234/v1   # LM Studio (Mac) or llama-server :8080 (Linux)
# in opencode.json: set the provider baseURL to http://127.0.0.1:8787/v1
# use OpenCode normally; Ctrl-C probe to get the session report
```

## `fit` command (milestone 1)
Inputs:
- Hardware, detected: total RAM; on macOS the Metal wired limit (`iogpu.wired_limit_mb`, 0 = default ~75% of RAM); on Linux `nvidia-smi --query-gpu=memory.total,name`; chip/GPU name for the bandwidth table. `--reserve-gb` (default 8 on unified memory, 2 for discrete VRAM) for OS + harness + browser.
- Model: a GGUF path (single or sharded), a directory (scan), or an MLX/safetensors directory (read `config.json`). With no argument, scan the LM Studio, Ollama, and `~/.cache/huggingface` model dirs.
- `--context` (default: table at 8k, 16k, 32k, 64k, 128k) and `--kv` (`f16` default, `q8_0`, `q4_0`).

Read from the GGUF header (no full-file read): architecture, `general.parameter_count` (else sum tensor shapes), `n_layer`, `n_head_kv`, head dim (`n_embd_head_k` else `n_embd / n_head`), `file_type` (quant), MoE `n_expert` / `n_expert_used`, and `general.base_model` / repo owner for the baseline check.

Compute (every value labeled `measured` or `inferred`):
- `weights_gb` = file size (measured).
- `kv_bytes_per_token` = 2 × n_layer × n_head_kv × head_dim × bytes(kv type) (inferred from header).
- `total_gb(context)` = weights + kv + ~0.5 GB compute buffer; `fits` = `yes` / `tight` (within 10% of budget) / `no` against `budget = memory − reserve`.
- `decode_tok_s_est` = chip bandwidth ÷ bytes of *active* weights per token (dense: all; MoE: `n_expert_used / n_expert` share + shared layers). Bandwidth comes from a small embedded table (M1–M5 Pro/Max/Ultra, RTX 3090/4090/5090); unknown chip → `--bandwidth-gb-s` required, else the column is omitted. Always labeled inferred; `capture` supplies the measured number.
- Baseline check: repo owner in an allowlist (vendor orgs, `unsloth`, `bartowski`, `lmstudio-community`) → `baseline`; otherwise `variant — not a baseline; compare against the base model before trusting it`. Ignore `mmproj-*` files, but report them as reclaimable disk.

Output: one table per model, plus the binding constraint in words, e.g. `27B Q4_K_S: fits at 32k (24.1 of 24 GB budget: tight), not at 64k. Binding constraint: unified memory; raising iogpu.wired_limit_mb would add ~4 GB.` `--json` emits the same with sources.

`--suggest`: print ≤5 baseline models for the memory class (≤16, 24–32, 48–64, 96–128 GB) from an embedded `models.yaml` with a mandatory `updated:` date printed in the header. The list is curation, not measurement, and is labeled as such; it is the only part of `fit` that decays.

## Architecture
- Go 1.22+, no cgo, static binaries for darwin-arm64 and linux-amd64. `go install` + curl installer.
- Reverse proxy on `127.0.0.1:8787`. Instruments `POST /v1/chat/completions`; passes every other path through untouched.
- Body is never buffered on the response side. SSE chunks are forwarded byte-exact as they arrive. Client cancel propagates upstream via request context.
- Request body is read once (needed for accounting), then forwarded unchanged — with one exception: if the request is `stream: true` and lacks `stream_options.include_usage`, inject it so the upstream reports `usage`. Record that this injection happened.

## Per-turn measurements
Every field carries a `source` of `measured` (reported by the upstream), `observed` (timed or counted by probe from bytes), or `inferred` (derived, e.g. prefix diff). Never upgrade a label.

Request decomposition (observed):
- `system_tokens`, `tools_tokens` (per tool name, sorted desc), `history_tokens` (prior user/assistant/tool messages), `current_user_tokens`, `request_bytes`, `tool_count`, `message_count`.
- Token counts: use upstream `usage.prompt_tokens` for the total (measured); apportion sub-totals with a local tokenizer estimate (inferred, tiktoken-compatible BPE; state which encoding). If usage is absent, the total is also inferred and the report says so.

Prefix stability (inferred):
- Serialize `[system, tools, messages...]` deterministically; keep the previous turn's serialization in memory only.
- `lcp_tokens` = longest common prefix with the previous turn; `re_prefilled_tokens = prompt_tokens - lcp_tokens`.
- `prefix_break`: the first divergent location — message index, role, field — and an 80-char redacted excerpt around it (e.g. `system[0] @ char 1412: "Current time: 22:41:0|3"`). This is the finding that converts the number into a fix.

Timing (observed):
- `ttft_ms` (request received → first content delta), `decode_ms`, `total_ms`, `output_tokens` (measured if usage present), `decode_tok_s`.
- `prefill_share = ttft_ms / total_ms`.

Backend telemetry (measured, optional):
- llama-server: `timings.prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms` from the response; `/slots` for cache state when reachable.
- LM Studio: per-response `stats` (`time_to_first_token`, `tokens_per_second`) when present.
- A cache hit/miss is only asserted from backend fields. Without them the report says "prefix changed at X; TTFT rose from A to B" — a delta, not a cause.

## Output
Live, one line per turn:
```
turn 7  prompt 51,204 (meas)  +harness 48,930  re-prefilled 50,811 (inf)  ttft 9.8s  prefill 91%  break: system[0]@1412 "Current time: …"
```
Session report on exit (also `--json` → NDJSON per turn + summary object):
1. Lead line: prefill share of turn latency (median, p90) and re-prefilled tokens per turn.
2. Harness-added tokens: system prompt, tools table (name, tokens, % of prompt).
3. Prefix stability: turns with a break, the top recurring break locations.
4. Backend telemetry if any, else a one-line note that cache reuse was not measurable.
5. Environment: OS, arch, upstream kind and version if discoverable, probe version, tokenizer used.

## Privacy
Default: counts, hashes, and redacted 80-char excerpts only. `--dump DIR` opts in to writing raw request/response bodies, with obvious secrets (`sk-`, `ghp_`, bearer headers) masked. Reports are meant to be pasted into GitHub issues.

## Edge cases to handle
`fit`:
- Sharded GGUF (`-00001-of-00003`): sum sizes, read the header from shard 1 only.
- Architectures with per-layer KV differences or missing keys: fall back to `n_embd / n_head` and say which key was missing; never print a number without a source.
- MoE models: active-parameter estimate for speed, full weights for memory.
- MLX / safetensors directories: `config.json` fields (`num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads`); weights from file sizes.
- Unknown chip or headless Linux without `nvidia-smi`: memory table still prints; speed column omitted with a one-line reason.
- Model dir symlinks and duplicate files (LM Studio + Ollama both holding the same GGUF): dedupe by size + header hash.

`capture`:
- Non-streaming requests; streaming with and without usage chunks; upstream that ignores `stream_options`.
- Upstream errors and non-JSON bodies: forward verbatim, log the turn as failed, do not crash.
- Client disconnect mid-stream: cancel upstream, record `aborted: true`.
- Tool-call turns with empty content (TTFT = first delta of any kind, note which).
- Multiple concurrent sessions: prefix state is keyed per client (remote port + `User-Agent`), with a flag to force single-session.
- Requests over `--max-body` (default 32 MiB): forward without accounting, mark `unaccounted`.
- Tokenizer mismatch: report the encoding; treat sub-totals as inferred.

## Validation paths (both before release)
- Mac (M1 Max, 32 GB): `fit` on the DavidAU Qwen 3.8 27B Q4_K_S already on disk must say `variant`, `16/64 layers hold KV`, fits at 64k, `no` at 256k. Met on 2026-09-17: the header's `full_attention_interval = 4` gives 65,536 KV bytes/token, and `fit`'s 16,384 MiB at 262k / 2,048 MiB at 32k match Ollama's server log to the MiB, which is why the 256k app default spilled 32 of 66 layers to CPU (0.35 tok/s) and 32k ran at 11.35 tok/s. Then OpenCode → LM Studio with `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL, plus the Nemotron 4B that is slow today.
- Linux/4090 (24 GB VRAM): `fit` must show the 27B Q4 fitting at 32k with `--reserve-gb 2` and the 125B Flash-Next as `no` at every context; then OpenCode → llama-server.
- Success: `fit`'s predicted fit/no-fit is confirmed by actually loading the model at that context on both machines, and on the ~50k-prefill machine the `capture` report names the harness-added tokens and the prefix break; the report is good enough to attach to an OpenCode issue as-is.

## Tests
- `fit` unit: GGUF header parser against synthetic fixture headers (dense, MoE, sharded, missing `n_head_kv`); KV and budget math with hand-checked expected values; baseline/variant classification; `models.yaml` schema (date, publisher allowlist, memory class) fails the build if stale >90 days.
- `fit` integration: a real small GGUF fixture (<50 MB) end to end, and the MLX `config.json` path.
- `capture` unit: request decomposition, tokenizer apportionment, LCP and break location (fixtures: identical prefix; timestamp in system prompt; reordered tools; appended message only).
- Golden SSE: recorded chunk streams from LM Studio and llama-server (with/without usage) replayed through a fake upstream; assert byte-exact pass-through, TTFT, and usage capture.
- Integration: fake upstream with controllable delays; assert prefill share and cancel propagation.
- Schema: NDJSON output validated against a JSON Schema that requires a `source` on every numeric field.
- A failing-first test for each edge case above.

## Milestones (solo, ~4 weeks)
1. `fit`: GGUF/MLX header readers, hardware detection, memory table, speed estimate, baseline check, `--suggest`, tests. Ship it alone as v0.1.0 — it is useful by itself and is what you need this week (week 1).
2. `capture`: proxy + streaming pass-through + timing + cancel (week 2).
3. Token accounting + prefix diff + live line + report + tests (week 3).
4. Backend telemetry, both validation paths, dogfood, README with a real `fit` table and a real `capture` report screenshot (week 4).

## v0.2 candidates (only if people re-run it)
`verify` (before/after a config change), `bench` (paired minimal vs captured request), `fit --hf org/repo` (read a Hugging Face config without downloading), Continue/Cline adapters, always-on mode with history.
