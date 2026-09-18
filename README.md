# probe

`probe` is a diagnostic for local coding-agent setups (OpenCode, Cline, Continue
talking to LM Studio, llama-server or Ollama). It tells you whether a model
fits your machine and how fast it will decode before you load it, and it
measures exactly what your harness adds to every prompt once it's running.

## Install

```
go install github.com/jraesly/reimagined-guacamole/cmd/probe@latest
```

or build from a checkout:

```
make build   # bin/probe
make check   # gofmt, vet, tests
```

Requires Go 1.22+.

## Quick start: `fit`

Given a model file (local or not yet downloaded), `fit` reads the GGUF/MLX
header, computes the KV-cache cost per token, and checks it against this
machine's memory at several context sizes:

```
$ probe fit --context 32k,64k,128k,256k
Machine   Apple M1 Max, 32 GB unified, 400 GB/s (table)
Budget    24.0 GB for weights + KV  [32.0 GB RAM minus 8 GB reserve for OS, harness and browser]
KV cache  f16

qwen3.8:27b  (qwen3.8:27b-32k)
  qwen35 dense Q4_K_M, 15.7 GB weights (measured), 27.3B params (observed), 17/65 layers hold KV
  baseline (library)
  context  KV GB  total GB  fits
  32k      2.1    18.3      yes
  64k      4.2    20.4      yes
  128k     8.5    24.7      no
  256k     17.0   33.2      no
  max context that fits: 64k (KV per token 69632 bytes, inferred from header)
  decode: 13.1 tok/s measured (ollama, 4k context, 2026-09-18)
  decode estimate: ~13 tok/s (inferred, medium confidence; calibrated from 1 measured dense run(s) on this machine)

qwen3.6:35b  (qwen3.6:35b-32k)
  qwen35moe MoE (~3.5B active per token, inferred) Q4_K_M, 20.2 GB weights (measured), 35.5B params (observed), 11/41 layers hold KV
  baseline (library)
  context  KV GB  total GB  fits
  32k      0.7    21.4      yes
  64k      1.4    22.1      tight
  128k     2.8    23.5      tight
  256k     5.5    26.2      no
  max context that fits: 128k (KV per token 22528 bytes, inferred from header)
  decode: 53.2 tok/s measured (ollama, 4k context, 2026-09-18)
```

This machine's own numbers: with the Ollama app's default 262k context, this
same model ran at **0.35 tok/s** with 34 of 66 layers pushed to CPU (a 16 GB
f16 KV cache at that context); capped at 32k it ran at **11.35 tok/s** with
all 66 layers on GPU. `fit` reproduces Ollama's KV allocation to the MiB,
because the header's `full_attention_interval = 4` means only 17 of 65 layers
hold a cache at all — the usual "0.25 MB/token" rule of thumb is 4x too high
here.

Other useful invocations:

```
probe fit --suggest                # short, dated list of baseline models for this memory class
probe fit --suggest --online       # ...and fit each suggestion from its registry header
probe fit ollama:laguna-xs-2.1     # a model you haven't pulled: header only, nothing downloaded
probe fit --measure                # run installed Ollama models briefly to calibrate estimates
probe fit --json ...               # every number carries a source label
probe header path/to/model.gguf    # dump the raw metadata fit reads
```

`--measure` at 4k context on this machine: `qwen3.8:27b` 13.1 tok/s,
`qwen3.6:35b` (MoE, ~3.5B active) 53.2 tok/s — recorded to
`~/.config/probe/calibration.json` and reused by future `fit` runs on any
model of the same kind (dense/MoE), in place of the built-in chip table.

See `docs/ESTIMATES.md` for exactly which figure comes from which
computation.

## Quick start: `capture`

`capture` is a session-scoped OpenAI-compatible proxy. Point your harness at
it instead of your model server; it forwards everything unchanged and prints
what the harness is actually sending:

```
$ probe capture --upstream http://127.0.0.1:11434/v1
probe capture 0.1.0-dev
  listening  http://127.0.0.1:8787/v1  →  http://127.0.0.1:11434/v1
  point your harness's OpenAI-compatible base URL at http://127.0.0.1:8787/v1 and use it normally
  request contents are not stored; press Ctrl-C for the session report

turn 1  prompt 425 (meas)  +harness 390  first turn  ttft 1.1s  prefill 52%
turn 2  prompt 448 (meas)  +harness 374  re-prefilled 41 (inf)  ttft 206ms  prefill 17%  prefix ok
turn 3  prompt 478 (meas)  +harness 356  re-prefilled 125 (inf)  ttft 915ms  prefill 40%  break: system[0].content@84 "po. Answer briefly. Current time: 22:41:|19⏎user:What does …"
^C
Prefill was 40% of turn latency (median; p90 52%) and the harness re-prefilled 41 tokens per turn (median) across 3 turns.
1 of 3 turns broke the prompt prefix.

Harness-added tokens
  system tokens (median): 48
  tools tokens (median): 326
  tool        max tokens  % of median prompt
  write_file  123         27.5%
  read_file   111         24.8%
  bash        105         23.5%
```

That is a real session against Ollama with a three-turn fake harness: turn 2
only appended a message and got its answer started in 206 ms; turn 3 changed
one timestamp in the system prompt, the cache had to be rebuilt from byte 84,
and time-to-first-token rose to 915 ms — 12.3% of the session's prompt tokens
were re-prefilled for nothing, and the report names the exact byte. `--json` writes NDJSON per turn plus a summary
object; `--report` writes the same session as a Markdown report meant to be
pasted into a GitHub issue as-is. See `docs/CAPTURE.md` for how every figure
is produced.

### Pointing OpenCode (or any OpenAI-compatible harness) at the proxy

Start `probe capture --upstream <your real server>`, then change the
harness's provider configuration so its base URL points at
`http://127.0.0.1:8787/v1` instead of the model server directly, keeping the
same model name and API key it already uses. Exactly which config key holds
that base URL is harness-specific (OpenCode, Cline and Continue each have
their own provider block) — check your harness's own provider docs for the
field name. Everything probe doesn't recognize (non-chat-completions paths,
non-POST requests) passes through untouched, so this is safe to leave in
place.

## Provenance labels

Every number `probe` prints carries one of three labels, and a report never
upgrades a label from a lower one to a higher one:

- **measured** — read directly from a file, from hardware, or reported by the
  upstream server (file sizes, `general.parameter_count`, a matched chip
  bandwidth, `usage.prompt_tokens`, llama.cpp/LM Studio timings).
- **observed** — counted or timed by probe itself (tensor element sums,
  wall-clock TTFT and decode time, calibration runs).
- **inferred** — derived by arithmetic or a heuristic (KV-cache byte math,
  the roofline speed estimate, character-apportioned token sub-totals,
  prefix-diff results).

## What it gets right that a rule of thumb does not

- **Hybrid KV rule**: architectures like the Qwen 3.5/3.6/3.8 family only keep
  a KV cache on every Nth layer (`full_attention_interval`, or an explicit
  recurrent-layer mask). `fit` reads that from the header and reproduces
  Ollama's allocation to the MiB.
- **MoE active bytes from tensor names**: the decode-speed estimate for a
  mixture-of-experts model comes from summing routed-expert tensors at
  `used/total` share plus always-read attention and shared-expert tensors —
  not from the marketing "A3B" label.
- **Tied embeddings**: when there's no separate `output.weight`, the input
  embedding table is the LM head and is read in full per token instead of
  being treated as a negligible row lookup.
- **Variant flagging**: a model's owner/repo is checked against an allowlist
  of vendor orgs and faithful-quant publishers; anything else is labeled a
  variant with a note to compare against the base model first.
- **Calibration over a chip table**: `fit --measure` records this machine's
  actual bytes-per-token vs. tokens/second and uses that median for every
  future estimate of the same kind (dense/MoE), instead of a single
  bandwidth number from a fixed table of chips.

## Privacy

`fit` reads only file headers (local files, or a few MB via HTTP range
requests for `ollama:`/`hf:` remote refs); nothing is downloaded or cached.

`capture` never stores request or response contents by default — only
counts, timings, hashes, and redacted 80-character excerpts around a prefix
break (with common secret shapes like `sk-`, `ghp_`, and bearer tokens
masked). Passing `--dump DIR` opts in to writing redacted raw bodies to disk
for local debugging; it is off unless you ask for it.

## Status and limitations

Milestones 1–3 of `docs/spec.md` are implemented: `fit` (header reading,
memory table, speed estimate, baseline check, `--suggest`, `--measure`
calibration) and `capture` (proxy, streaming pass-through, token accounting,
prefix diff, live line, session report). See `docs/spec.md` for what's left
(backend-telemetry validation on a second, Linux/4090 machine) and
`docs/spec.md#implemented-deviations-from-this-spec` for where the shipped
tool differs from the original plan.

Known limitations:

- Architectures with multi-head latent attention (DeepSeek-style
  `kv_lora_rank`) or sliding-window/shared-KV caps are tagged and warned
  about, but the KV figure is an **upper bound** — a `no` verdict for these
  may be pessimistic.
- AMD/ROCm GPUs are not detected or modeled; only macOS Metal and
  NVIDIA/`nvidia-smi` are.
- `fit --measure` calibrates against installed **Ollama** models only; LM
  Studio cannot be measured this way yet.
- The `--suggest` list is curated, not benchmarked by probe — it's a dated
  pointer to baseline models, not a ranking.
- The chip bandwidth table (used when no calibration exists) covers a fixed
  set of chips and GPUs at a single bin; unlisted hardware needs
  `--bandwidth-gb-s` or gets no speed estimate at all.

## Further reading

- `docs/spec.md` — the v0.1 spec and what's implemented against it.
- `docs/ESTIMATES.md` — exactly how `fit` computes every number.
- `docs/CAPTURE.md` — exactly how `capture` computes and labels every figure.
