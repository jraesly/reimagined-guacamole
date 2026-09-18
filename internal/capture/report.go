// Reporting for probe capture: a one-line live update per turn, a terminal
// summary, a Markdown report for pasting into a GitHub issue, and NDJSON.
//
// Every number follows the convention from internal/report: it carries a
// Source, and prose says "measured" when the server told us, "observed" when
// probe's own clock or counters produced it, and "inferred" when it was
// derived by arithmetic or a heuristic (character-based apportionment, the
// four-chars-per-token estimate, or anything downstream of those).
package capture

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// Env is the run-level context main.go supplies for the summary and
// Markdown reports; it is not per-turn data.
type Env struct {
	Version  string
	OS, Arch string
	Upstream string
	Listen   string
	Started  time.Time
	Ended    time.Time
}

// LiveLine is printed after every turn, one line, no trailing newline.
func LiveLine(t Turn) string {
	if t.Status != "ok" {
		return liveLineNotOK(t)
	}
	parts := []string{
		fmt.Sprintf("turn %d", t.Seq),
		fmt.Sprintf("prompt %s (%s)", commas(t.PromptTokens.Value), srcAbbrev(t.PromptTokens.Source)),
		fmt.Sprintf("+harness %s", commas(t.HarnessTokens.Value)),
	}
	if t.FirstTurn {
		parts = append(parts, "first turn")
	} else {
		parts = append(parts, fmt.Sprintf("re-prefilled %s (%s)", commas(t.ReprefilledTokens.Value), srcAbbrev(t.ReprefilledTokens.Source)))
	}
	parts = append(parts,
		fmt.Sprintf("ttft %s", formatDuration(t.TTFTms.Value)),
		fmt.Sprintf("prefill %d%%", int(math.Round(t.PrefillShare.Value*100))),
	)
	if !t.FirstTurn {
		if t.PrefixBreak != nil {
			br := t.PrefixBreak
			parts = append(parts, fmt.Sprintf("break: %s@%d %q", br.Segment, br.Offset, truncateExcerpt(br.Excerpt, 60)))
		} else {
			parts = append(parts, "prefix ok")
		}
	}
	return strings.Join(parts, "  ")
}

func liveLineNotOK(t Turn) string {
	switch t.Status {
	case "upstream-error":
		line := fmt.Sprintf("turn %d  upstream-error %d", t.Seq, t.HTTPStatus)
		if t.Error != "" {
			line += "  " + t.Error
		}
		return line
	case "unaccounted":
		return fmt.Sprintf("turn %d  unaccounted (%s body)", t.Seq, humanBytes(t.RequestBytes))
	case "aborted":
		line := fmt.Sprintf("turn %d  aborted", t.Seq)
		if t.Error != "" {
			line += "  " + t.Error
		}
		return line
	default:
		return fmt.Sprintf("turn %d  %s", t.Seq, t.Status)
	}
}

func srcAbbrev(s model.Source) string {
	if s == model.Measured {
		return "meas"
	}
	return "inf"
}

func formatDuration(ms float64) string {
	if ms < 1000 {
		return fmt.Sprintf("%.0fms", ms)
	}
	return fmt.Sprintf("%.1fs", ms/1000)
}

func humanBytes(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1f MB", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1f KB", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func truncateExcerpt(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	if max <= 1 {
		return string(r[:max])
	}
	return string(r[:max-1]) + "…"
}

// commas renders a token count (possibly fractional, since inferred counts
// are apportioned by characters) with thousands separators.
func commas(v float64) string {
	n := int64(math.Round(v))
	neg := n < 0
	if neg {
		n = -n
	}
	s := strconv.FormatInt(n, 10)
	var out []byte
	for i := 0; i < len(s); i++ {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, s[i])
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}

// percentile uses the nearest-rank method on values already sorted ascending.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(math.Ceil(p / 100 * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

func median(vals []float64) float64 { return percentile(sortedCopy(vals), 50) }

func sortedCopy(vals []float64) []float64 {
	c := append([]float64(nil), vals...)
	sort.Float64s(c)
	return c
}

// toolRow is one row of the harness-added tools table.
type toolRow struct {
	Name              string
	MaxTokens         float64
	PctOfMedianPrompt float64
}

// breakGroup is one Segment's worth of prefix breaks across the session.
type breakGroup struct {
	Segment string
	Count   int
	Example string
}

// llamaTelemetry summarizes llama.cpp timings across turns.
type llamaTelemetry struct {
	MedianPromptMs   float64
	MedianTokS       float64
	HasCache         bool
	MedianCacheRatio float64
}

// lmstudioTelemetry summarizes LM Studio stats across turns.
type lmstudioTelemetry struct {
	MedianTokS   float64
	MedianTTFTms float64
}

// summary is everything WriteSummary and WriteMarkdown need, computed once
// so the two renderers can't drift.
type summary struct {
	turnCount int

	// lead
	skipPrefixLead    bool // every turn is FirstTurn, or there's only one turn
	leadSkipNote      string
	prefillMedianPct  int
	prefillP90Pct     int
	reprefilledMedian float64
	brokenCount       int

	// harness-added tokens
	medianSystemTokens float64
	medianToolsTokens  float64
	toolRows           []toolRow

	// prefix stability
	breakGroups      []breakGroup
	totalReprefilled float64
	totalPrompt      float64
	wastedPct        float64

	// server telemetry
	hasBackend bool
	llama      *llamaTelemetry
	lmstudio   *lmstudioTelemetry

	// environment
	env              Env
	usageInjectedAny bool
}

func buildSummary(turns []Turn, env Env) summary {
	s := summary{turnCount: len(turns), env: env}

	allFirst := true
	for _, t := range turns {
		if !t.FirstTurn {
			allFirst = false
		}
		if t.UsageInjected {
			s.usageInjectedAny = true
		}
	}
	if len(turns) <= 1 || allFirst {
		s.skipPrefixLead = true
		if len(turns) <= 1 {
			s.leadSkipNote = fmt.Sprintf("Only %d turn was captured; there is no session-level prefix behavior to summarize.", len(turns))
		} else {
			s.leadSkipNote = fmt.Sprintf("Every one of the %d captured turns started a new session (first turn); there is no repeated-prefix behavior to summarize.", len(turns))
		}
	}

	var prefillShares, reprefilled, promptTokens, systemTokens, toolsTokens []float64
	toolMax := map[string]float64{}
	breakBySeg := map[string]*breakGroup{}
	var breakOrder []string

	for _, t := range turns {
		prefillShares = append(prefillShares, t.PrefillShare.Value*100)
		promptTokens = append(promptTokens, t.PromptTokens.Value)
		systemTokens = append(systemTokens, t.SystemTokens.Value)
		toolsTokens = append(toolsTokens, t.ToolsTokens.Value)
		for _, pt := range t.PerTool {
			if pt.Tokens.Value > toolMax[pt.Name] {
				toolMax[pt.Name] = pt.Tokens.Value
			}
		}
		if !t.FirstTurn {
			reprefilled = append(reprefilled, t.ReprefilledTokens.Value)
			s.totalReprefilled += t.ReprefilledTokens.Value
			if t.PrefixBreak != nil {
				s.brokenCount++
				g, ok := breakBySeg[t.PrefixBreak.Segment]
				if !ok {
					g = &breakGroup{Segment: t.PrefixBreak.Segment, Example: t.PrefixBreak.Excerpt}
					breakBySeg[t.PrefixBreak.Segment] = g
					breakOrder = append(breakOrder, t.PrefixBreak.Segment)
				}
				g.Count++
			}
		}
		s.totalPrompt += t.PromptTokens.Value
		if t.Backend != nil {
			s.hasBackend = true
		}
	}

	if len(prefillShares) > 0 {
		s.prefillMedianPct = int(math.Round(median(prefillShares)))
		s.prefillP90Pct = int(math.Round(percentile(sortedCopy(prefillShares), 90)))
	}
	s.reprefilledMedian = median(reprefilled)
	s.medianSystemTokens = median(systemTokens)
	s.medianToolsTokens = median(toolsTokens)

	medianPrompt := median(promptTokens)
	for name, max := range toolMax {
		row := toolRow{Name: name, MaxTokens: max}
		if medianPrompt > 0 {
			row.PctOfMedianPrompt = max / medianPrompt * 100
		}
		s.toolRows = append(s.toolRows, row)
	}
	sort.Slice(s.toolRows, func(i, j int) bool {
		if s.toolRows[i].MaxTokens != s.toolRows[j].MaxTokens {
			return s.toolRows[i].MaxTokens > s.toolRows[j].MaxTokens
		}
		return s.toolRows[i].Name < s.toolRows[j].Name
	})
	if len(s.toolRows) > 15 {
		s.toolRows = s.toolRows[:15]
	}

	for _, seg := range breakOrder {
		s.breakGroups = append(s.breakGroups, *breakBySeg[seg])
	}
	sort.SliceStable(s.breakGroups, func(i, j int) bool { return s.breakGroups[i].Count > s.breakGroups[j].Count })
	if len(s.breakGroups) > 10 {
		s.breakGroups = s.breakGroups[:10]
	}
	if s.totalPrompt > 0 {
		s.wastedPct = s.totalReprefilled / s.totalPrompt * 100
	}

	if s.hasBackend {
		var promptMs, tokS, cacheRatio []float64
		for _, t := range turns {
			b := t.Backend
			if b == nil || b.Source != "llama.cpp timings" {
				continue
			}
			if b.PromptMs != nil {
				promptMs = append(promptMs, *b.PromptMs)
			}
			if b.PredictedN != nil && b.PredictedMs != nil && *b.PredictedMs > 0 {
				tokS = append(tokS, *b.PredictedN / *b.PredictedMs * 1000)
			}
			if b.CacheN != nil && b.PromptN != nil && *b.PromptN > 0 {
				cacheRatio = append(cacheRatio, *b.CacheN / *b.PromptN)
			}
		}
		if len(promptMs) > 0 || len(tokS) > 0 {
			s.llama = &llamaTelemetry{MedianPromptMs: median(promptMs), MedianTokS: median(tokS)}
			if len(cacheRatio) > 0 {
				s.llama.HasCache = true
				s.llama.MedianCacheRatio = median(cacheRatio)
			}
		}

		var lmTokS, lmTTFT []float64
		for _, t := range turns {
			b := t.Backend
			if b == nil || b.Source != "lmstudio stats" {
				continue
			}
			if b.TokensPerSecond != nil {
				lmTokS = append(lmTokS, *b.TokensPerSecond)
			}
			if b.TimeToFirstToken != nil {
				lmTTFT = append(lmTTFT, *b.TimeToFirstToken)
			}
		}
		if len(lmTokS) > 0 || len(lmTTFT) > 0 {
			s.lmstudio = &lmstudioTelemetry{MedianTokS: median(lmTokS), MedianTTFTms: median(lmTTFT)}
		}
	}

	return s
}

// WriteSummary renders the terminal-facing session summary.
func WriteSummary(w io.Writer, turns []Turn, env Env) {
	if len(turns) == 0 {
		fmt.Fprintln(w, "No turns captured.")
		return
	}
	s := buildSummary(turns, env)

	if s.skipPrefixLead {
		fmt.Fprintln(w, s.leadSkipNote)
	} else {
		fmt.Fprintf(w, "Prefill was %d%% of turn latency (median; p90 %d%%) and the harness re-prefilled %s tokens per turn (median) across %d turns.\n",
			s.prefillMedianPct, s.prefillP90Pct, commas(s.reprefilledMedian), s.turnCount)
		fmt.Fprintf(w, "%d of %d turns broke the prompt prefix.\n", s.brokenCount, s.turnCount)
	}

	fmt.Fprintln(w, "\nHarness-added tokens")
	fmt.Fprintf(w, "  system tokens (median): %s\n  tools tokens (median): %s\n", commas(s.medianSystemTokens), commas(s.medianToolsTokens))
	if len(s.toolRows) > 0 {
		tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  tool\tmax tokens\t% of median prompt")
		for _, r := range s.toolRows {
			fmt.Fprintf(tw, "  %s\t%s\t%.1f%%\n", r.Name, commas(r.MaxTokens), r.PctOfMedianPrompt)
		}
		tw.Flush()
	}
	fmt.Fprintln(w, "  (sub-totals are apportioned by characters, inferred)")

	fmt.Fprintln(w, "\nPrefix stability")
	if len(s.breakGroups) > 0 {
		tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  segment\tbreaks\texample")
		for _, g := range s.breakGroups {
			fmt.Fprintf(tw, "  %s\t%d\t%s\n", g.Segment, g.Count, truncateExcerpt(g.Example, 60))
		}
		tw.Flush()
	} else if !s.skipPrefixLead {
		fmt.Fprintln(w, "  no breaks observed")
	}
	fmt.Fprintf(w, "  wasted prefill: %s of %s prompt tokens re-prefilled (%.1f%%, inferred)\n",
		commas(s.totalReprefilled), commas(s.totalPrompt), s.wastedPct)

	fmt.Fprintln(w, "\nServer telemetry")
	writeTelemetryText(w, s)

	fmt.Fprintln(w, "\nEnvironment")
	writeEnvText(w, s)
}

func writeTelemetryText(w io.Writer, s summary) {
	if !s.hasBackend {
		fmt.Fprintln(w, "  Cache reuse was not measurable: the server reported no timings; prefix figures above are inferred from request text.")
		return
	}
	if s.llama != nil {
		fmt.Fprintf(w, "  llama.cpp: prompt_ms %.0f (median), %.1f tok/s predicted (median)\n", s.llama.MedianPromptMs, s.llama.MedianTokS)
		if s.llama.HasCache {
			fmt.Fprintf(w, "  cache-hit ratio (cache_n / prompt_n): %.1f%% (measured, median)\n", s.llama.MedianCacheRatio*100)
		}
	}
	if s.lmstudio != nil {
		fmt.Fprintf(w, "  lmstudio: %.1f tok/s (median), time to first token %s (median)\n", s.lmstudio.MedianTokS, formatDuration(s.lmstudio.MedianTTFTms))
	}
}

func writeEnvText(w io.Writer, s summary) {
	e := s.env
	fmt.Fprintf(w, "  probe %s  %s/%s\n", e.Version, e.OS, e.Arch)
	fmt.Fprintf(w, "  upstream %s\n  listen %s\n", e.Upstream, e.Listen)
	fmt.Fprintf(w, "  started %s\n  ended %s\n", e.Started.Format(time.RFC3339), e.Ended.Format(time.RFC3339))
	fmt.Fprintf(w, "  usage injected on any turn: %s\n", yesno(s.usageInjectedAny))
}

func yesno(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// WriteMarkdown renders a GitHub-issue-ready report with the same content
// as WriteSummary.
func WriteMarkdown(w io.Writer, turns []Turn, env Env) {
	if len(turns) == 0 {
		fmt.Fprintln(w, "# probe capture report")
		fmt.Fprintln(w, "\nNo turns captured.")
		return
	}
	s := buildSummary(turns, env)

	fmt.Fprintln(w, "# probe capture report")
	fmt.Fprintln(w)
	if s.skipPrefixLead {
		fmt.Fprintln(w, s.leadSkipNote)
	} else {
		fmt.Fprintf(w, "Prefill was %d%% of turn latency (median; p90 %d%%) and the harness re-prefilled %s tokens per turn (median) across %d turns. %d of %d turns broke the prompt prefix.\n",
			s.prefillMedianPct, s.prefillP90Pct, commas(s.reprefilledMedian), s.turnCount, s.brokenCount, s.turnCount)
	}

	fmt.Fprintln(w, "\n## Harness-added tokens")
	fmt.Fprintf(w, "\nSystem tokens (median): %s. Tools tokens (median): %s.\n", commas(s.medianSystemTokens), commas(s.medianToolsTokens))
	if len(s.toolRows) > 0 {
		fmt.Fprintln(w)
		mdTableHeader(w, "tool", "max tokens", "% of median prompt")
		for _, r := range s.toolRows {
			mdTableRow(w, r.Name, commas(r.MaxTokens), fmt.Sprintf("%.1f%%", r.PctOfMedianPrompt))
		}
	}
	fmt.Fprintln(w, "\nSub-totals are apportioned by characters (inferred).")

	fmt.Fprintln(w, "\n## Prefix stability")
	if len(s.breakGroups) > 0 {
		fmt.Fprintln(w)
		mdTableHeader(w, "segment", "breaks", "example")
		for _, g := range s.breakGroups {
			mdTableRow(w, g.Segment, fmt.Sprintf("%d", g.Count), "`"+truncateExcerpt(g.Example, 60)+"`")
		}
	} else if !s.skipPrefixLead {
		fmt.Fprintln(w, "\nNo breaks observed.")
	}
	fmt.Fprintf(w, "\nWasted prefill: %s of %s prompt tokens re-prefilled (%.1f%%, inferred).\n",
		commas(s.totalReprefilled), commas(s.totalPrompt), s.wastedPct)

	fmt.Fprintln(w, "\n## Server telemetry")
	fmt.Fprintln(w)
	writeTelemetryMarkdown(w, s)

	fmt.Fprintln(w, "\n## Environment")
	fmt.Fprintln(w)
	writeEnvMarkdown(w, s)

	fmt.Fprintf(w, "\nGenerated by probe %s; every number is labeled measured, observed or inferred; request contents are not included.\n", env.Version)
}

func writeTelemetryMarkdown(w io.Writer, s summary) {
	if !s.hasBackend {
		fmt.Fprintln(w, "Cache reuse was not measurable: the server reported no timings; prefix figures above are inferred from request text.")
		return
	}
	if s.llama != nil {
		fmt.Fprintf(w, "- llama.cpp: prompt_ms %.0f (median), %.1f tok/s predicted (median)\n", s.llama.MedianPromptMs, s.llama.MedianTokS)
		if s.llama.HasCache {
			fmt.Fprintf(w, "- cache-hit ratio (cache_n / prompt_n): %.1f%% (measured, median)\n", s.llama.MedianCacheRatio*100)
		}
	}
	if s.lmstudio != nil {
		fmt.Fprintf(w, "- lmstudio: %.1f tok/s (median), time to first token %s (median)\n", s.lmstudio.MedianTokS, formatDuration(s.lmstudio.MedianTTFTms))
	}
}

func writeEnvMarkdown(w io.Writer, s summary) {
	e := s.env
	fmt.Fprintf(w, "- probe %s, %s/%s\n", e.Version, e.OS, e.Arch)
	fmt.Fprintf(w, "- upstream %s, listen %s\n", e.Upstream, e.Listen)
	fmt.Fprintf(w, "- started %s, ended %s\n", e.Started.Format(time.RFC3339), e.Ended.Format(time.RFC3339))
	fmt.Fprintf(w, "- usage injected on any turn: %s\n", yesno(s.usageInjectedAny))
}

func mdTableHeader(w io.Writer, cols ...string) {
	fmt.Fprintln(w, "| "+strings.Join(cols, " | ")+" |")
	seps := make([]string, len(cols))
	for i := range seps {
		seps[i] = "---"
	}
	fmt.Fprintln(w, "| "+strings.Join(seps, " | ")+" |")
}

func mdTableRow(w io.Writer, cols ...string) {
	fmt.Fprintln(w, "| "+strings.Join(cols, " | ")+" |")
}

// WriteNDJSON emits one JSON-encoded Turn per line.
func WriteNDJSON(w io.Writer, turns []Turn) error {
	enc := json.NewEncoder(w)
	for _, t := range turns {
		if err := enc.Encode(t); err != nil {
			return err
		}
	}
	return nil
}
