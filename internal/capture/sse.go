package capture

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Usage is the OpenAI usage block.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// Timings is llama.cpp's server telemetry, present when the upstream is
// llama-server (or LM Studio's llama.cpp engine exposes it).
type Timings struct {
	PromptN     *float64 `json:"prompt_n"`
	PromptMs    *float64 `json:"prompt_ms"`
	PredictedN  *float64 `json:"predicted_n"`
	PredictedMs *float64 `json:"predicted_ms"`
	CacheN      *float64 `json:"cache_n"`
}

// Stats is LM Studio's per-response block.
type Stats struct {
	TokensPerSecond  *float64 `json:"tokens_per_second"`
	TimeToFirstToken *float64 `json:"time_to_first_token"`
}

// event is one parsed chat-completion chunk or full response.
type event struct {
	Usage   *Usage   `json:"usage"`
	Timings *Timings `json:"timings"`
	Stats   *Stats   `json:"stats"`
	Choices []struct {
		Delta *struct {
			Content   *string         `json:"content"`
			Reasoning *string         `json:"reasoning_content"` // llama-server, LM Studio
			Thinking  *string         `json:"reasoning"`         // Ollama
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
		Message *struct {
			Content   *string         `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

// hasOutput reports whether the event carries any generated text, including
// reasoning/thinking tokens.
func (e *event) hasOutput() bool {
	if e.hasContent() {
		return true
	}
	for _, c := range e.Choices {
		if c.Delta != nil && ((c.Delta.Reasoning != nil && *c.Delta.Reasoning != "") ||
			(c.Delta.Thinking != nil && *c.Delta.Thinking != "")) {
			return true
		}
	}
	return false
}

// hasContent reports whether the event carries visible answer content or a
// tool call, i.e. something the harness acts on, as opposed to reasoning.
func (e *event) hasContent() bool {
	for _, c := range e.Choices {
		if c.Delta != nil && ((c.Delta.Content != nil && *c.Delta.Content != "") || hasToolCalls(c.Delta.ToolCalls)) {
			return true
		}
		if c.Message != nil && ((c.Message.Content != nil && *c.Message.Content != "") || hasToolCalls(c.Message.ToolCalls)) {
			return true
		}
	}
	return false
}

// hasToolCalls is true for a non-empty JSON array; "null", "[]" and
// whitespace variants are not output.
func hasToolCalls(raw json.RawMessage) bool {
	var calls []json.RawMessage
	return json.Unmarshal(raw, &calls) == nil && len(calls) > 0
}

// maxLine bounds the pending SSE line. A stream that never sends a newline
// (or a pathological single event) must not grow memory without limit; the
// bytes are still relayed, only the observation is dropped.
const maxLine = 1 << 20

// Observation is what the response parser learned from a stream or body.
type Observation struct {
	FirstOutput   bool // set once any generated token (including reasoning) was seen
	FirstContent  bool // set once visible content or a tool call was seen
	Chunks        int  // SSE data events seen
	Usage         *Usage
	Timings       *Timings
	Stats         *Stats
	Done          bool
	ParseFailures int
}

func (o *Observation) absorb(e *event) {
	if e.Usage != nil {
		o.Usage = e.Usage
	}
	if e.Timings != nil {
		o.Timings = e.Timings
	}
	if e.Stats != nil {
		o.Stats = e.Stats
	}
}

// sseScanner feeds raw bytes and emits parsed events without owning the
// stream: the proxy copies bytes to the client and calls Write with the same
// bytes, so nothing the harness sees is altered.
type sseScanner struct {
	buf            bytes.Buffer
	obs            *Observation
	onFirst        func() // first generated token of any kind
	onFirstContent func() // first visible content or tool call
	dropping       bool   // inside an oversized line: skip bytes until the next newline
}

// Write consumes a chunk of the SSE byte stream.
func (s *sseScanner) Write(p []byte) (int, error) {
	for len(p) > 0 {
		if s.dropping {
			nl := bytes.IndexByte(p, '\n')
			if nl < 0 {
				return len(p), nil
			}
			p = p[nl+1:]
			s.dropping = false
			continue
		}
		nl := bytes.IndexByte(p, '\n')
		if nl < 0 {
			if s.buf.Len()+len(p) > maxLine {
				s.buf.Reset()
				s.dropping = true
				s.obs.ParseFailures++
				return len(p), nil
			}
			s.buf.Write(p)
			return len(p), nil
		}
		s.buf.Write(p[:nl])
		line := s.buf.String()
		s.buf.Reset()
		p = p[nl+1:]
		s.line(strings.TrimRight(line, "\r"))
	}
	return 0, nil
}

func (s *sseScanner) line(l string) {
	data, ok := strings.CutPrefix(l, "data:")
	if !ok {
		return
	}
	data = strings.TrimSpace(data)
	if data == "[DONE]" {
		s.obs.Done = true
		return
	}
	s.obs.Chunks++
	var e event
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		s.obs.ParseFailures++
		return
	}
	s.obs.absorb(&e)
	if !s.obs.FirstOutput && e.hasOutput() {
		s.obs.FirstOutput = true
		if s.onFirst != nil {
			s.onFirst()
		}
	}
	if !s.obs.FirstContent && e.hasContent() {
		s.obs.FirstContent = true
		if s.onFirstContent != nil {
			s.onFirstContent()
		}
	}
}

// parseBody handles a non-streaming JSON response.
func parseBody(body []byte, obs *Observation) {
	var e event
	if err := json.Unmarshal(body, &e); err != nil {
		obs.ParseFailures++
		return
	}
	obs.absorb(&e)
	obs.FirstOutput = e.hasOutput()
	obs.FirstContent = e.hasContent()
	obs.Done = true
}
