// Package report renders fit results as a terminal table or JSON. Every
// number in the JSON carries its source label.
package report

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/jraesly/reimagined-guacamole/internal/fit"
	"github.com/jraesly/reimagined-guacamole/internal/hw"
	"github.com/jraesly/reimagined-guacamole/internal/model"
	"github.com/jraesly/reimagined-guacamole/internal/suggest"
)

// Value is a number with provenance.
type Value struct {
	Value  float64      `json:"value"`
	Source model.Source `json:"source"`
}

// ModelResult is the fit outcome for one model.
type ModelResult struct {
	Name           string      `json:"name"`
	Aliases        []string    `json:"aliases,omitempty"`
	Path           string      `json:"path"`
	Arch           string      `json:"arch"`
	Quant          string      `json:"quant"`
	Params         Value       `json:"params"`
	ActiveParams   Value       `json:"active_params"`
	WeightsGB      Value       `json:"weights_gb"`
	KVBytesPerTok  Value       `json:"kv_bytes_per_token"`
	AttentionLayer int         `json:"attention_layers"`
	Layers         uint32      `json:"layers"`
	MoE            bool        `json:"moe"`
	Baseline       bool        `json:"baseline"`
	BaselineNote   string      `json:"baseline_note"`
	Remote         bool        `json:"remote"` // header read from a registry; weights not on disk
	Rows           []fit.Row   `json:"rows"`
	MaxContext     uint64      `json:"max_context"`
	DecodeTokS     *SpeedValue `json:"decode_tok_s,omitempty"`
	Warnings       []string    `json:"warnings,omitempty"`
	Extras         []string    `json:"reclaimable_files,omitempty"`
	Error          string      `json:"error,omitempty"`
}

// SpeedValue is an estimated rate with confidence.
type SpeedValue struct {
	Value      float64      `json:"value"`
	Source     model.Source `json:"source"`
	Confidence string       `json:"confidence"`
}

// Report is the whole fit output.
type Report struct {
	Hardware   hw.Info        `json:"hardware"`
	Budget     hw.Budget      `json:"budget"`
	KV         fit.KVType     `json:"kv_type"`
	Models     []ModelResult  `json:"models"`
	Suggestion *suggest.Class `json:"suggestion,omitempty"`
	ListDate   string         `json:"suggestion_list_updated,omitempty"`
	ListNote   string         `json:"suggestion_note,omitempty"`
}

// Build assembles a ModelResult from a model and the fit table.
func Build(m *model.Model, aliases []string, extras []string, budgetGB float64, opts fit.Options, bandwidth float64) ModelResult {
	r := ModelResult{
		Name: m.Name, Aliases: aliases, Path: m.Path, Arch: m.Arch, Quant: m.Quant,
		Params:         Value{float64(m.Params), m.ParamsSource},
		ActiveParams:   Value{float64(m.ActiveParams()), model.Inferred},
		WeightsGB:      Value{float64(m.WeightsBytes) / fit.GiB, model.Measured},
		AttentionLayer: m.AttentionLayers(), Layers: m.Layers, MoE: m.IsMoE(),
		Baseline: m.Baseline, BaselineNote: m.BaselineNote, Remote: m.Remote, Warnings: m.Warnings, Extras: extras,
	}
	per, err := fit.KVBytesPerToken(m, opts.KV)
	if err != nil {
		r.Error = err.Error()
		return r
	}
	r.KVBytesPerTok = Value{per, model.Inferred}
	rows, err := fit.Table(m, budgetGB, opts)
	if err != nil {
		r.Error = err.Error()
		return r
	}
	r.Rows = rows
	r.MaxContext, _ = fit.MaxContext(rows)
	if s, ok := fit.DecodeEstimate(m, bandwidth); ok {
		r.DecodeTokS = &SpeedValue{s.TokPerSec, model.Inferred, s.Confidence}
	}
	return r
}

// WriteJSON emits the report as indented JSON.
func WriteJSON(w io.Writer, r Report) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// WriteText renders the human-readable report.
func WriteText(w io.Writer, r Report) {
	h := r.Hardware
	memory := fmt.Sprintf("%.0f GB unified", h.RAMGB)
	if !h.Unified {
		memory = fmt.Sprintf("%.0f GB RAM, %.1f GB VRAM", h.RAMGB, h.VRAMGB)
	}
	bw := "unknown bandwidth"
	if h.BandwidthKnown {
		bw = fmt.Sprintf("%.0f GB/s (table)", h.BandwidthGBs)
	}
	fmt.Fprintf(w, "Machine   %s, %s, %s\n", h.Chip, memory, bw)
	fmt.Fprintf(w, "Budget    %.1f GB for weights + KV  [%s]\n", r.Budget.GB, r.Budget.Reason)
	fmt.Fprintf(w, "KV cache  %s\n", r.KV)
	for _, n := range h.Notes {
		fmt.Fprintf(w, "Note      %s\n", n)
	}
	for _, m := range r.Models {
		fmt.Fprintln(w)
		name := m.Name
		if len(m.Aliases) > 0 {
			name += "  (" + strings.Join(m.Aliases, ", ") + ")"
		}
		if m.Remote {
			name += fmt.Sprintf("  [not downloaded: %.1f GB pull from %s]", m.WeightsGB.Value, m.Path)
		}
		fmt.Fprintf(w, "%s\n", name)
		if m.Error != "" {
			fmt.Fprintf(w, "  error: %s\n", m.Error)
			continue
		}
		shape := "dense"
		if m.MoE {
			shape = fmt.Sprintf("MoE (~%.1fB active per token, inferred)", m.ActiveParams.Value/1e9)
		}
		fmt.Fprintf(w, "  %s %s %s, %.1f GB weights (%s), %.1fB params (%s), %d/%d layers hold KV\n",
			m.Arch, shape, m.Quant, m.WeightsGB.Value, m.WeightsGB.Source, m.Params.Value/1e9, m.Params.Source, m.AttentionLayer, m.Layers)
		fmt.Fprintf(w, "  %s\n", m.BaselineNote)
		tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  context\tKV GB\ttotal GB\tfits")
		for _, row := range m.Rows {
			fmt.Fprintf(tw, "  %s\t%.1f\t%.1f\t%s\n", ctxLabel(row.Context), row.KVGB, row.TotalGB, row.Verdict)
		}
		tw.Flush()
		if m.MaxContext > 0 {
			fmt.Fprintf(w, "  max context that fits: %s (KV per token %.0f bytes, inferred from header)\n", ctxLabel(m.MaxContext), m.KVBytesPerTok.Value)
		} else {
			fmt.Fprintf(w, "  does not fit at any listed context\n")
		}
		if m.DecodeTokS != nil {
			fmt.Fprintf(w, "  decode estimate: ~%.0f tok/s (inferred, %s confidence; run capture for a measured number)\n", m.DecodeTokS.Value, m.DecodeTokS.Confidence)
		}
		for _, x := range m.Extras {
			fmt.Fprintf(w, "  reclaimable: %s (vision projector, not needed for coding)\n", x)
		}
		for _, wmsg := range m.Warnings {
			fmt.Fprintf(w, "  warning: %s\n", wmsg)
		}
	}
	if r.Suggestion != nil {
		fmt.Fprintf(w, "\nSuggested baselines for %s (curated list dated %s; not measured)\n", r.Suggestion.Name, r.ListDate)
		for _, e := range r.Suggestion.Models {
			fmt.Fprintf(w, "  %-24s %-14s %s\n      %s\n", e.Name, e.Shape, e.Why, e.Pull)
		}
		if r.ListNote != "" {
			fmt.Fprintf(w, "  %s\n", r.ListNote)
		}
	}
}

func ctxLabel(n uint64) string {
	if n%1024 == 0 {
		return fmt.Sprintf("%dk", n/1024)
	}
	return fmt.Sprintf("%d", n)
}
