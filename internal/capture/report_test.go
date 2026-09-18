package capture

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// newTurn returns a minimal "ok" turn with sane zero-ish defaults; tests
// override only the fields they care about.
func newTurn(seq int) Turn {
	return Turn{
		Seq:     seq,
		Session: "s1",
		At:      time.Date(2026, 9, 17, 12, 0, 0, 0, time.UTC),
		Model:   "test-model",
		Stream:  true,
		Status:  "ok",
	}
}

func f64(v float64) *float64 { return &v }

// ---- LiveLine ----

func TestLiveLineMeasuredOkTurnWithBreak(t *testing.T) {
	tu := newTurn(7)
	tu.PromptTokens = Value{51204, model.Measured}
	tu.HarnessTokens = Value{48930, model.Inferred}
	tu.ReprefilledTokens = Value{50811, model.Inferred}
	tu.TTFTms = Value{9800, model.Observed}
	tu.PrefillShare = Value{0.91, model.Inferred}
	tu.PrefixBreak = &Break{Segment: "system[0].content", Offset: 1412, Excerpt: "Current time: 22:41:0|3", Reason: "changed"}

	want := `turn 7  prompt 51,204 (meas)  +harness 48,930  re-prefilled 50,811 (inf)  ttft 9.8s  prefill 91%  break: system[0].content@1412 "Current time: 22:41:0|3"`
	if got := LiveLine(tu); got != want {
		t.Errorf("LiveLine =\n%q\nwant\n%q", got, want)
	}
}

func TestLiveLineFirstTurn(t *testing.T) {
	tu := newTurn(1)
	tu.FirstTurn = true
	tu.PromptTokens = Value{5000, model.Inferred}
	tu.HarnessTokens = Value{2000, model.Inferred}
	tu.TTFTms = Value{500, model.Observed}
	tu.PrefillShare = Value{0.4, model.Inferred}

	got := LiveLine(tu)
	if !strings.Contains(got, "first turn") {
		t.Errorf("LiveLine = %q, want %q substring", got, "first turn")
	}
	if strings.Contains(got, "re-prefilled") || strings.Contains(got, "break:") || strings.Contains(got, "prefix ok") {
		t.Errorf("LiveLine for first turn should omit prefix fields: %q", got)
	}
}

func TestLiveLinePrefixOkInferredPrompt(t *testing.T) {
	tu := newTurn(2)
	tu.PromptTokens = Value{3000, model.Inferred}
	tu.HarnessTokens = Value{1000, model.Inferred}
	tu.ReprefilledTokens = Value{500, model.Inferred}
	tu.TTFTms = Value{640, model.Observed}
	tu.PrefillShare = Value{0.33, model.Inferred}
	// PrefixBreak left nil: prefix ok

	got := LiveLine(tu)
	if !strings.Contains(got, "prompt 3,000 (inf)") {
		t.Errorf("LiveLine = %q, want inferred prompt tokens", got)
	}
	if !strings.Contains(got, "prefix ok") {
		t.Errorf("LiveLine = %q, want \"prefix ok\"", got)
	}
	if !strings.Contains(got, "ttft 640ms") {
		t.Errorf("LiveLine = %q, want sub-second ttft as ms", got)
	}
}

func TestLiveLineUpstreamError(t *testing.T) {
	tu := newTurn(3)
	tu.Status = "upstream-error"
	tu.HTTPStatus = 500

	want := "turn 3  upstream-error 500"
	if got := LiveLine(tu); got != want {
		t.Errorf("LiveLine = %q, want %q", got, want)
	}
}

func TestLiveLineUnaccounted(t *testing.T) {
	tu := newTurn(4)
	tu.Status = "unaccounted"
	tu.RequestBytes = 2_100_000

	want := "turn 4  unaccounted (2.1 MB body)"
	if got := LiveLine(tu); got != want {
		t.Errorf("LiveLine = %q, want %q", got, want)
	}
}

func TestLiveLineAborted(t *testing.T) {
	tu := newTurn(5)
	tu.Status = "aborted"

	want := "turn 5  aborted"
	if got := LiveLine(tu); got != want {
		t.Errorf("LiveLine = %q, want %q", got, want)
	}
}

func TestTruncateExcerptAt60Chars(t *testing.T) {
	long := strings.Repeat("a", 80)
	got := truncateExcerpt(long, 60)
	if r := []rune(got); len(r) != 60 {
		t.Fatalf("truncateExcerpt length = %d, want 60", len(r))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("truncateExcerpt = %q, want ellipsis suffix", got)
	}

	short := "short excerpt"
	if got := truncateExcerpt(short, 60); got != short {
		t.Errorf("truncateExcerpt should not modify short strings: %q", got)
	}

	tu := newTurn(6)
	tu.PromptTokens = Value{1000, model.Measured}
	tu.PrefillShare = Value{0.5, model.Inferred}
	tu.TTFTms = Value{100, model.Observed}
	tu.PrefixBreak = &Break{Segment: "message[3].content", Offset: 10, Excerpt: long}
	line := LiveLine(tu)
	if strings.Contains(line, strings.Repeat("a", 61)) {
		t.Errorf("LiveLine excerpt not truncated: %q", line)
	}
}

// ---- WriteSummary / WriteMarkdown fixtures ----

// fiveTurnSession builds a hand-computed fixture:
//
//	prefill shares (%): 50, 60, 70, 80, 90        -> median 70, p90 90
//	reprefilled tokens (turns 2-5): 1000,2000,3000,4000 -> median 2000
//	prompt tokens: 10000,20000,30000,40000,50000  -> median 30000
//	breaks: turn3 & turn4 "system[0].content", turn5 "tools[0] (read_file)"
//	turn2 has no break ("prefix ok")               -> 3 of 5 broke
//	tools: read_file max 500, write_file max 9000  -> write_file first
func fiveTurnSession() []Turn {
	mk := func(seq int, first bool, prefillPct float64, prompt float64, reprefilled float64) Turn {
		tu := newTurn(seq)
		tu.FirstTurn = first
		tu.PrefillShare = Value{prefillPct / 100, model.Inferred}
		tu.PromptTokens = Value{prompt, model.Measured}
		tu.ReprefilledTokens = Value{reprefilled, model.Inferred}
		tu.SystemTokens = Value{200, model.Inferred}
		tu.ToolsTokens = Value{300, model.Inferred}
		return tu
	}
	t1 := mk(1, true, 50, 10000, 0)
	t2 := mk(2, false, 60, 20000, 1000)
	t2.PerTool = []ToolTokens{{Name: "read_file", Tokens: Value{300, model.Inferred}}}
	t3 := mk(3, false, 70, 30000, 2000)
	t3.PrefixBreak = &Break{Segment: "system[0].content", Offset: 5, Excerpt: "Current time: 09:00:0|1", Reason: "changed"}
	t3.PerTool = []ToolTokens{{Name: "read_file", Tokens: Value{500, model.Inferred}}, {Name: "write_file", Tokens: Value{9000, model.Inferred}}}
	t4 := mk(4, false, 80, 40000, 3000)
	t4.PrefixBreak = &Break{Segment: "system[0].content", Offset: 5, Excerpt: "Current time: 09:00:1|2", Reason: "changed"}
	t5 := mk(5, false, 90, 50000, 4000)
	t5.PrefixBreak = &Break{Segment: "tools[0] (read_file)", Offset: 0, Excerpt: "reordered tool", Reason: "reordered"}
	return []Turn{t1, t2, t3, t4, t5}
}

func testEnv() Env {
	return Env{
		Version:  "0.1.0-test",
		OS:       "darwin",
		Arch:     "arm64",
		Upstream: "http://127.0.0.1:8080",
		Listen:   "127.0.0.1:4000",
		Started:  time.Date(2026, 9, 17, 12, 0, 0, 0, time.UTC),
		Ended:    time.Date(2026, 9, 17, 12, 5, 0, 0, time.UTC),
	}
}

func TestWriteSummaryLeadLineAndBreakGrouping(t *testing.T) {
	var buf bytes.Buffer
	WriteSummary(&buf, fiveTurnSession(), testEnv())
	out := buf.String()

	wantLead := "Prefill was 70% of turn latency (median; p90 90%) and the harness re-prefilled 2,000 tokens per turn (median) across 5 turns."
	if !strings.Contains(out, wantLead) {
		t.Errorf("summary missing lead line.\nwant substring: %s\ngot:\n%s", wantLead, out)
	}
	if !strings.Contains(out, "3 of 5 turns broke the prompt prefix.") {
		t.Errorf("summary missing break count line, got:\n%s", out)
	}

	// break grouping: system[0].content (count 2) must appear before
	// tools[0] (read_file) (count 1).
	iSys := strings.Index(out, "system[0].content")
	iTools := strings.Index(out, "tools[0] (read_file)")
	if iSys == -1 || iTools == -1 || iSys > iTools {
		t.Errorf("break groups not sorted desc by count: sys@%d tools@%d\n%s", iSys, iTools, out)
	}
	if !strings.Contains(out, "system[0].content") || !strings.Contains(out, "2") {
		t.Errorf("expected system[0].content break count of 2 in:\n%s", out)
	}
}

func TestWriteSummaryToolsTableSortedDesc(t *testing.T) {
	var buf bytes.Buffer
	WriteSummary(&buf, fiveTurnSession(), testEnv())
	out := buf.String()

	iWrite := strings.Index(out, "write_file")
	iRead := strings.Index(out, "read_file")
	if iWrite == -1 || iRead == -1 || iWrite > iRead {
		t.Errorf("expected write_file (max 9000) before read_file (max 500) in tools table:\n%s", out)
	}
}

func TestWriteSummaryNoBackendNotMeasurable(t *testing.T) {
	var buf bytes.Buffer
	WriteSummary(&buf, fiveTurnSession(), testEnv())
	out := buf.String()
	if !strings.Contains(out, "Cache reuse was not measurable") {
		t.Errorf("expected not-measurable note without Backend, got:\n%s", out)
	}
}

func TestWriteSummaryLlamaCppCacheHitRatioMeasured(t *testing.T) {
	turns := fiveTurnSession()
	turns[2].Backend = &Backend{
		Source:  "llama.cpp timings",
		PromptN: f64(1000), PromptMs: f64(500),
		PredictedN: f64(100), PredictedMs: f64(200),
		CacheN: f64(800),
	}
	var buf bytes.Buffer
	WriteSummary(&buf, turns, testEnv())
	out := buf.String()
	if !strings.Contains(out, "cache-hit ratio") {
		t.Errorf("expected cache-hit ratio line, got:\n%s", out)
	}
	if !strings.Contains(out, "measured") {
		t.Errorf("expected cache-hit ratio labeled measured, got:\n%s", out)
	}
	if !strings.Contains(out, "80.0%") {
		t.Errorf("expected cache-hit ratio of 80.0%%, got:\n%s", out)
	}
}

func TestWriteSummaryOnlyFirstTurnsSaysSo(t *testing.T) {
	turns := []Turn{
		func() Turn { tu := newTurn(1); tu.FirstTurn = true; return tu }(),
		func() Turn { tu := newTurn(2); tu.FirstTurn = true; return tu }(),
	}
	var buf bytes.Buffer
	WriteSummary(&buf, turns, testEnv())
	out := buf.String()
	if strings.Contains(out, "broke the prompt prefix") {
		t.Errorf("all-first-turn session should not report prefix breaks:\n%s", out)
	}
	if !strings.Contains(out, "first turn") {
		t.Errorf("expected an explanatory note about first turns, got:\n%s", out)
	}
}

func TestWriteSummaryEmptyTurns(t *testing.T) {
	var buf bytes.Buffer
	WriteSummary(&buf, nil, testEnv())
	if !strings.Contains(strings.ToLower(buf.String()), "no turns captured") {
		t.Errorf("expected no-turns message, got: %q", buf.String())
	}
}

func TestWriteMarkdownEmptyTurns(t *testing.T) {
	var buf bytes.Buffer
	WriteMarkdown(&buf, nil, testEnv())
	if !strings.Contains(strings.ToLower(buf.String()), "no turns captured") {
		t.Errorf("expected no-turns message, got: %q", buf.String())
	}
}

func TestWriteMarkdownHasHeadingsTableAndFooter(t *testing.T) {
	var buf bytes.Buffer
	WriteMarkdown(&buf, fiveTurnSession(), testEnv())
	out := buf.String()

	if !strings.Contains(out, "## Harness-added tokens") {
		t.Errorf("missing markdown heading, got:\n%s", out)
	}
	if !strings.Contains(out, "| --- |") {
		t.Errorf("missing pipe table separator, got:\n%s", out)
	}
	wantFooter := "Generated by probe 0.1.0-test; every number is labeled measured, observed or inferred; request contents are not included."
	if !strings.Contains(out, wantFooter) {
		t.Errorf("missing footer line, got:\n%s", out)
	}
}

// ---- WriteNDJSON ----

func TestWriteNDJSONRoundTripsAndOmitsNilBreak(t *testing.T) {
	turns := []Turn{newTurn(1), newTurn(2), newTurn(3)}
	turns[1].PrefixBreak = &Break{Segment: "system[0].content", Offset: 3, Excerpt: "x|y", Reason: "changed"}

	var buf bytes.Buffer
	if err := WriteNDJSON(&buf, turns); err != nil {
		t.Fatalf("WriteNDJSON error: %v", err)
	}
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3", len(lines))
	}
	for i, line := range lines {
		var got Turn
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("line %d did not unmarshal: %v\n%s", i, err, line)
		}
		if got.Seq != turns[i].Seq {
			t.Errorf("line %d Seq = %d, want %d", i, got.Seq, turns[i].Seq)
		}
	}
	if strings.Contains(lines[0], "prefix_break") {
		t.Errorf("turn with nil PrefixBreak should omit the key: %s", lines[0])
	}
	if !strings.Contains(lines[1], "prefix_break") {
		t.Errorf("turn with PrefixBreak should include the key: %s", lines[1])
	}
}
