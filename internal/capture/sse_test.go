package capture

import (
	"testing"
)

func TestSSEScannerDetectsFirstOutputUsageAndDone(t *testing.T) {
	obs := &Observation{}
	firsts := 0
	s := &sseScanner{obs: obs, onFirst: func() { firsts++ }}
	stream := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":41,\"completion_tokens\":2},\"timings\":{\"prompt_n\":41,\"prompt_ms\":120.5,\"cache_n\":30}}\n\n" +
		"data: [DONE]\n\n"
	// Feed in awkward chunk boundaries to exercise line buffering.
	for i := 0; i < len(stream); i += 7 {
		end := i + 7
		if end > len(stream) {
			end = len(stream)
		}
		_, _ = s.Write([]byte(stream[i:end]))
	}
	if firsts != 1 || !obs.FirstOutput {
		t.Errorf("first output fired %d times", firsts)
	}
	if obs.Chunks != 5 || !obs.Done || obs.ParseFailures != 0 {
		t.Errorf("obs = %+v", obs)
	}
	if obs.Usage == nil || obs.Usage.PromptTokens != 41 || obs.Timings == nil || *obs.Timings.CacheN != 30 {
		t.Errorf("usage/timings = %+v %+v", obs.Usage, obs.Timings)
	}
}

func TestSSEScannerToolCallCountsAsOutput(t *testing.T) {
	obs := &Observation{}
	s := &sseScanner{obs: obs}
	_, _ = s.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\"}}]}}]}\n"))
	if !obs.FirstOutput {
		t.Error("tool_calls delta should count as first output")
	}
	for _, notOutput := range []string{
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[]}}]}\n",
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":null}}]}\n",
		"data: {\"choices\":[{\"delta\":{\"tool_calls\": [ ] }}]}\n",
		"data: {\"choices\":[{\"message\":{}}]}\n",
	} {
		o := &Observation{}
		sc := &sseScanner{obs: o}
		_, _ = sc.Write([]byte(notOutput))
		if o.FirstOutput {
			t.Errorf("%q should not count as output", notOutput)
		}
	}
}

func TestSSEScannerBoundsUnterminatedLines(t *testing.T) {
	obs := &Observation{}
	s := &sseScanner{obs: obs}
	chunk := make([]byte, 64<<10)
	for i := range chunk {
		chunk[i] = 'x'
	}
	for i := 0; i < 40; i++ { // 2.5 MiB with no newline
		_, _ = s.Write(chunk)
		if s.buf.Len() > maxLine {
			t.Fatalf("pending line grew to %d bytes", s.buf.Len())
		}
	}
	if !s.dropping || obs.ParseFailures != 1 {
		t.Errorf("oversized line not dropped: dropping=%v failures=%d", s.dropping, obs.ParseFailures)
	}
	// After the newline the scanner recovers and parses the next event.
	_, _ = s.Write([]byte("tail\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n"))
	if s.dropping || !obs.FirstOutput || obs.Chunks != 1 {
		t.Errorf("scanner did not recover: %+v dropping=%v", obs, s.dropping)
	}
}

func TestSSEScannerIgnoresNonDataLinesAndBadJSON(t *testing.T) {
	obs := &Observation{}
	s := &sseScanner{obs: obs}
	_, _ = s.Write([]byte(": keepalive\nevent: ping\ndata: {not json}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n"))
	if obs.ParseFailures != 1 || obs.Chunks != 2 || !obs.FirstOutput {
		t.Errorf("obs = %+v", obs)
	}
}

func TestSSEScannerOllamaReasoningKeyIsOutput(t *testing.T) {
	// Ollama's OpenAI-compatible endpoint streams thinking as "reasoning"
	// with an empty "content"; a role-only first chunk must not count.
	obs := &Observation{}
	s := &sseScanner{obs: obs}
	_, _ = s.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n"))
	if obs.FirstOutput {
		t.Error("role-only delta counted as output")
	}
	_, _ = s.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"\",\"reasoning\":\"Here\"}}]}\n"))
	if !obs.FirstOutput {
		t.Error("Ollama reasoning delta should count as first output")
	}
}

func TestParseBodyNonStreaming(t *testing.T) {
	obs := &Observation{}
	parseBody([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":1},"stats":{"tokens_per_second":12.5}}`), obs)
	if !obs.FirstOutput || !obs.Done || obs.Usage.PromptTokens != 10 || obs.Stats == nil || *obs.Stats.TokensPerSecond != 12.5 {
		t.Errorf("obs = %+v", obs)
	}
	bad := &Observation{}
	parseBody([]byte("<html>"), bad)
	if bad.ParseFailures != 1 || bad.Done {
		t.Errorf("bad = %+v", bad)
	}
}
