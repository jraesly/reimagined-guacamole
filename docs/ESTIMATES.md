# How `probe fit` computes its estimates

Every number in a `probe fit` report maps to a computation in `internal/model`,
`internal/fit`, `internal/hw` and `internal/calib`. This note says which figure
comes from where, so a reader can trust it or challenge it.

## Provenance labels

`model.Source` (internal/model/model.go) has three values, in decreasing
trust:

- `measured` — read directly from the file or hardware (file sizes, `general.parameter_count`, a matched chip-name bandwidth).
- `observed` — counted or timed by probe (tensor element sums when `parameter_count` is missing).
- `inferred` — derived by arithmetic or heuristic (anything using a fraction, a quantization table, or an efficiency factor).

A report must never upgrade a label. Tensor-derived read counts are always
`inferred`: storage size is a proxy for the bytes a runtime transfers per
token, not a measurement of execution.

## Memory

- **Weights bytes** (`Model.WeightsBytes`) is the sum of on-disk file sizes in
  bytes, across all shards (`FromGGUF`). For a remote model it is the size the
  host reports. Memory figures are then converted with `GiB = 1024³`, so a
  "16.9 GB" file contributes ~15.8 GiB.
- **KV cache per token** (fit.go:KVBytesPerToken) is the sum over layers with
  KV heads of `heads × (key_len + value_len) × bytes_per_element`, where
  `bytes_per_element` is `2` for `f16`, `34/32` for `q8_0` and `18/32` for
  `q4_0` (block quants carry a per-block scale). `f16` is the default
  (`Table`). `key_len` comes from `attention.key_length`, else
  `embedding_length / head_count`; `value_length` defaults to `key_len`.
- **Hybrid architectures** (`model.go:applyHybridLayers`), following
  llama.cpp's precedence: an explicit `{arch}.attention.recurrent_layers`
  mask wins; otherwise `{arch}.full_attention_interval`; otherwise the
  architecture default (4 for `qwen35`, `qwen35moe`, `qwen3next`). Only every
  `interval`-th main layer keeps a KV cache; the trailing
  `nextn_predict_layers` keep theirs (Ollama's server log allocates one for
  them). Recurrent layers get `KVHeads[i] = 0`. Their small recurrent-state
  buffer is not modelled separately; the compute buffer is meant to cover it.
- **Missing `head_count_kv`** means plain multi-head attention and defaults to
  `head_count`, as llama.cpp does.
- **Caches the formula does not model**: multi-head latent attention
  (`attention.kv_lora_rank`, DeepSeek-style) stores a compressed cache, and
  sliding-window or shared-KV layers cap theirs at the window. For both, the
  model is tagged (`CacheModel` = `mla` / `swa`), a warning is printed, and the
  KV figure is an upper bound — a `no` verdict may be pessimistic.
- **Model maximum context**: `{arch}.context_length` is read and any table row
  beyond it is annotated, since a server defaulting to the model maximum is
  exactly how a 27B model ended up spilling to CPU on a 32 GB machine.
- **Compute buffer**: `Options.ComputeGB`, default `0.5` GB of scratch space on
  top of weights and KV.
- **Budget** (hw.go:Budget): unified memory takes the smaller of
  `RAM − reserve` and the wired limit (`iogpu.wired_limit_mb`, or 75% of RAM by
  default). Discrete GPUs take `VRAM − reserve`. CPU-only Linux takes
  `RAM − reserve`. Default reserve is 8 GB unified, 2 GB discrete.
- **Verdict** (fit.go:Classify): `total ≤ 0.9 × budget` → `yes`; `≤ budget` →
  `tight` (within 10%); otherwise `no`. Total is `weights + KV + compute`.

## Speed

- **Bytes read per token** (`Model.ReadBytes`) is built from each tensor's ggml
  type and shape, with rows padded to whole quantization blocks the way ggml
  stores them. Routed experts (`_exps.`) count `used/total` of the time;
  attention and shared experts (`_shexp.`) always read; the input embedding
  (`token_embd.*`) is a row lookup and excluded — unless there is no
  `output.weight`, in which case the embeddings are tied and the table is read
  in full as the LM head. If any tensor type is unrecognised, `ReadBytes` is
  zero and `BytesPerToken` falls back to `weights × ActiveFraction`
  (`inferred`). For dense models `ActiveFraction` is 1 and every byte counts.
  A remote model whose header covers only the first shard never contributes
  tensor sums; its expert fraction falls back to `used/total`.
- **Roofline** (fit.go:DecodeEstimate): `bandwidth_gb_s / (bytes_per_token / 1e9)`.
- **Default efficiencies** (fit.go, comment above `denseEfficiency`/`moeEfficiency`),
  calibrated on an Apple M1 Max (400 GB/s) with Ollama 0.34 on 2026-09-16 at
  32k context: dense `0.48` — Qwen3.8-27B Q4_K_M, 16.9 GB weights and a 23.7
  roofline, measured 11.35 tok/s; moe `0.125` — Qwen3.6-35B-A3B, 21.7 GB
  weights, ~3.5B of 35.5B params active (≈2.14 GB/token), roofline 187,
  measured 23.5 tok/s.
- **Why MoE is low confidence**: MoE decode is bounded by more than bandwidth
  — routing, small matmuls, per-expert overhead — so the factor is much lower
  and less transferable. Confidence on the table basis is `medium` dense /
  `low` MoE.
- **`fit --measure` calibration** (calib.go): each measured run records
  `bytes_per_token` and `tok_per_sec`; `effective GB/s =
  bytes_per_token / 1e9 × tok_per_sec`. `File.Effective` returns the *median*
  across samples of a kind. `Estimate` uses the calibration whenever the kind has
  at least one sample (tok/s = `eff / (bytes_per_token / 1e9)`), even if the
  bandwidth table is empty; otherwise it falls back to the table. Calibrated
  confidence is `medium`, rising to `high` only for dense models backed by at
  least two measured runs; the basis string names how many runs.

## Worked example: Qwen3.8-27B on an M1 Max

Header facts (fit_test.go:qwen38): 64 main layers with full attention every 4th
→ 16 attention layers; 4 KV heads × 256 key/256 value; 16.9 GB weights.

- KV: `16 × 4 × (256 + 256) × 2 = 65536 bytes/token`. At 32k context that is
  2048 MiB; at 256k, 16384 MiB — matching Ollama's log lines.
- Budget: Metal reported 25559 MiB total; minus the runtime's ~1 GB leaves
  ≈24 GB.
- Dense bytes/token = 16.9e9. Roofline `400 / 16.9 = 23.7`; `× 0.48 = 11.4`,
  inside tolerance of the measured 11.35 tok/s.
- Total at 32k: `15.8 + 2 + 0.5 = 18.3 GiB` → yes on 24 GB. At 256k:
  `15.8 + 16 + 0.5 = 32.3 GiB` → no.

## Known limitations

- The bandwidth table (hw.go) covers a fixed set of chips with single vendor
  figures for the common bin; chips that ship in lower-bandwidth bins, and
  chips not in the table (no speed estimate unless `--bandwidth-gb-s` is
  passed), are not handled. Multi-GPU Linux uses the first device only.
- MoE runtime overheads are not modelled — the `0.125` factor is a single
  heuristic fitted to one model.
- The bits-per-weight table (model.go) is used only when
  `general.parameter_count` is absent and tensors are incomplete; otherwise
  parameter count is read or counted directly.
- Tokenizer-free assumptions: no token budget, and the input embedding is
  treated as a negligible one-row read (unless tied to the output head).
- No thermal or sustained-bandwidth modelling; no timing/verification of the
  roofline on the target machine beyond `fit --measure` calibration.
- A `yes` verdict means estimated memory capacity under the runtime
  assumptions above (f16 KV unless told otherwise, one sequence, 0.5 GB
  scratch); it does not prove the runtime will load the model at that
  context. The verdict for MLA and sliding-window models is an upper bound.
- Partial GPU offload, batching and multi-GPU splits are not modelled; the
  parser assumes little-endian GGUF.