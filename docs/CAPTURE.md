# How `probe capture` measures a turn

`capture` is a reverse proxy on `127.0.0.1:8787` that sits between your
coding harness and its real model server (`--upstream`). It forwards every
request byte-exact and instruments only `POST .../chat/completions`;
everything else — including non-POST requests — passes through untouched.

## Session key

Requests are grouped into sessions by `User-Agent + model`, or by the
`X-Probe-Session` request header plus model when a harness sets it — use that
header when one harness runs several agents in parallel under one
User-Agent, so their prefix histories stay apart (the header is stripped
before the request reaches the upstream). Each session
keeps the previous turn's canonicalized prompt in memory (nothing else) so
the next turn can be diffed against it. Two harnesses hitting the proxy
concurrently, or the same harness switching models, get separate prefix
histories; there is no way to force them into one session key. A request
whose body can't be parsed as chat-completions JSON, or that exceeds
`--max-body` (default 32 MiB), is forwarded but marked `unaccounted` and
never touches session state.

## The `include_usage` injection

Most local servers only report `usage` (prompt/completion token counts) on
non-streaming responses. If a request has `"stream": true` without
`stream_options.include_usage`, `capture` adds that field before forwarding
— nothing else in the body changes. This is part of the OpenAI streaming API
that well-behaved OpenAI-compatible clients already tolerate: it asks the
server to emit one extra usage-only chunk at the end of the stream, which a
client that doesn't look for it simply ignores. The turn's `usage_injected`
flag records whenever this happened, so you can see whether the number you
measured came from the server directly or was elicited by probe.

## Statuses

Every turn ends in exactly one of:

- **ok** — forwarded and fully accounted.
- **upstream-error** — the upstream returned a non-2xx status, or the
  connection to it failed after the request started.
- **aborted** — the client disconnected before the response finished; probe
  cancels the upstream request via the same context.
- **unaccounted** — the body didn't parse as chat-completions JSON, or
  exceeded `--max-body`; it was still forwarded, just not measured.

## What is and is not written to disk

By default: nothing. The live line and session summary are the only output,
to the terminal. `--json FILE` writes one JSON object per turn (NDJSON) plus
a summary object; `--report FILE` writes the same session as Markdown.
Neither includes request or response bodies — only counts, timings, hashes,
and an 80-character redacted excerpt around a prefix break. `--dump DIR` is
the one opt-in that writes raw request/response bodies to disk, with common
secret shapes (`sk-…`, `ghp_…`, `gho_…`, Slack tokens, AWS access keys,
bearer headers, PEM private keys) masked before the write.

## Reading a report

1. **Lead line**: prefill share of turn latency — the fraction of total turn
   time spent before the first output token — as a median and p90 across the
   session, plus the median number of tokens re-prefilled per turn. This is
   the number that answers "is my harness wasting time re-processing the
   prompt." A session where every turn was a first turn (no repeat prefix to
   compare) skips this line and says so instead of printing a meaningless
   number.
2. **Harness-added tokens**: the median system-prompt and tools token counts,
   plus a table of tool definitions by name, their max size across the
   session, and that as a percentage of the median prompt. This is what your
   harness costs you before you type anything.
3. **Prefix stability**: which segments of the prompt (a system message, a
   tool definition, a history message) broke the shared prefix most often,
   with an example excerpt, plus **wasted prefill** — the total re-prefilled
   tokens as a percentage of total prompt tokens across the session. A live
   run through Ollama with a timestamp embedded in the system prompt showed
   `break: system[0].content@84 "…Current time: 22:41:|19"` and 12.3% wasted
   prefill; that one field is usually the actionable fix.
4. **Server telemetry**: llama.cpp's `timings` (`prompt_ms`, predicted
   tok/s, and a cache-hit ratio when `cache_n` is present) or LM Studio's
   `stats` block, when the upstream reports either. Without them, the report
   says plainly that cache reuse wasn't measurable and that the prefix
   figures above are inferred from request text, not confirmed by the
   server.

## Known limits

- **No tokenizer.** Only the upstream's own `usage.prompt_tokens` (when
  present) is a measured count. Every sub-total — system tokens, tools
  tokens, per-tool tokens, re-prefilled tokens — is apportioned from that
  total by character share of a canonicalized prompt string, not counted by
  a real tokenizer. If the upstream reports no usage at all, even the total
  falls back to a ~4-characters-per-token estimate and is labeled inferred.
- **Cache hits are only claimed with backend telemetry.** Without
  `cache_n`/`timings` from the server, probe reports a prefix-diff delta
  ("prefix changed at X; TTFT rose") — never a cause. It will not tell you
  the server actually reused its KV cache unless the server says so.
- **The prefix comparison is template-dependent.** `capture` canonicalizes
  `[tools..., messages...]` in that order because most chat templates inject
  tools before the conversation, but the real template is server- and
  model-specific, so a "break" is an inferred approximation of what the
  server's actual templated prompt did, not a byte-for-byte replay of it.
- **One session key per `User-Agent + model`.** If your harness reuses the
  same User-Agent and model across unrelated conversations, they'll be
  diffed against each other's prefixes; if it varies its User-Agent, each
  variation starts a fresh "first turn" with no prefix history.
