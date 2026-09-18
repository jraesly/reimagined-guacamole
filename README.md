# probe

Diagnostics for local coding-agent setups. v0.1 ships one command, `fit`:
given this machine and the model files already on it, say whether each model
fits, at what context, and roughly how fast — from arithmetic on the file
header, not from opinion.

```
probe fit                          # scans LM Studio, Ollama and the Hugging Face cache
probe fit --suggest                # short, dated list of baseline models for this memory class
probe fit --suggest --online       # ...and fits each suggestion from its registry header
probe fit ollama:laguna-xs-2.1     # a model you have not pulled: header only, no download
probe fit hf:unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
probe fit --context 32k,64k --kv q8_0 path/to/model.gguf
probe fit --json ...               # every number carries a source label
probe header path/to/model.gguf    # dump the metadata fit reads
```

Remote targets read a few megabytes of GGUF header (or `config.json` for MLX
repos) through HTTP range requests from the Ollama registry or Hugging Face,
plus the file size from the manifest. Nothing is downloaded or cached.

Every value is labeled `measured` (read from the file or hardware), `observed`
(counted by probe) or `inferred` (arithmetic or heuristic). The speed estimate
is inferred from memory bandwidth and is only a starting point; a measured
number needs a real run.

## Why

Model catalogs are chronological dumps, and the same model can run 30x slower
than it should because of one server setting (a 256k default context on a
32 GB machine puts half the layers on the CPU). `fit` reads the GGUF header,
computes the KV cache for each context size against the memory the GPU can
actually use, and flags community fine-tunes so you compare against a baseline
first.

## Build

Requires Go 1.22+.

```
make check   # gofmt, vet, tests
make build   # bin/probe
```

## What it gets right that a rule of thumb does not

Hybrid architectures (the Qwen 3.5/3.6/3.8 family) only keep a KV cache on
every 4th layer; `fit` reads `full_attention_interval` from the header and
reproduces Ollama's allocation to the MiB, where the usual "0.25 MB per token"
guess is 4x too high. MoE active parameters come from the expert tensor
names, not the marketing "A3B". Community fine-tunes are flagged as variants.

## Status

Milestone 1 of the v0.1 spec (`docs/spec.md`). `capture` (the harness
overhead proxy) comes next.
