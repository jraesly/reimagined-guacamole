// Package report renders fit results as a terminal table or JSON. Every
// number in the JSON carries its source label.
package report

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/jraesly/reimagined-guacamole/internal/calib"
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
	Tasks          []string    `json:"tasks,omitempty"`
	Remote         bool        `json:"remote"`                 // header read from a registry; weights not on disk
	InstalledAs    string      `json:"installed_as,omitempty"` // a remote hit whose weights are already on disk under this name
	Rows           []fit.Row   `json:"rows"`
	MaxContext     uint64      `json:"max_context"`
	DecodeTokS     *SpeedValue `json:"decode_tok_s,omitempty"`
	MeasuredTokS   *Measured   `json:"measured_tok_s,omitempty"`
	Warnings       []string    `json:"warnings,omitempty"`
	Extras         []string    `json:"reclaimable_files,omitempty"`
	Error          string      `json:"error,omitempty"`
}

// SpeedValue is an estimated rate with confidence and the basis used.
type SpeedValue struct {
	Value      float64      `json:"value"`
	Source     model.Source `json:"source"`
	Confidence string       `json:"confidence"`
	Basis      string       `json:"basis,omitempty"`
}

// Measured is a decode rate this machine actually produced.
type Measured struct {
	Value      float64      `json:"value"`
	Source     model.Source `json:"source"`
	Backend    string       `json:"backend"`
	Context    int          `json:"context"`
	MeasuredAt string       `json:"measured_at"`
}

// Calibration is the subset of calib.File the report needs.
type Calibration interface {
	fit.Calibration
	Lookup(names ...string) (calib.Sample, bool)
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
	SearchTask string         `json:"search_task,omitempty"`
	SearchHits []SearchHit    `json:"search_hits,omitempty"`
}

// SearchHit is one online search result; fitted ones also appear in Models.
type SearchHit struct {
	Ref          string   `json:"ref"`
	Name         string   `json:"name"`
	Source       string   `json:"source"`
	Description  string   `json:"description,omitempty"`
	Downloads    int64    `json:"downloads"`
	Capabilities []string `json:"capabilities,omitempty"`
}

// WriteSuggestions prints the curated list and the online search hits
// that follow a compact table.
func WriteSuggestions(w io.Writer, r Report) {
	if r.Suggestion != nil {
		fmt.Fprintf(w, "\nCurated baselines for %s (list dated %s; not measured)\n", r.Suggestion.Name, r.ListDate)
		for _, e := range r.Suggestion.Models {
			fmt.Fprintf(w, "  %-24s %-14s %s\n      %s\n", e.Name, e.Shape, e.Why, e.Pull)
		}
	}
	if len(r.SearchHits) > 0 {
		fmt.Fprintf(w, "\nOnline search for %s (Hugging Face + ollama.com; baselines unless --variants; fitted hits appear in the table above)\n", r.SearchTask)
		tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  ref\tsource\tdownloads\tcapabilities\tdescription")
		for _, h := range r.SearchHits {
			desc := h.Description
			if len(desc) > 70 {
				desc = desc[:67] + "…"
			}
			fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\n", h.Ref, h.Source, commasInt(h.Downloads), strings.Join(h.Capabilities, ","), desc)
		}
		tw.Flush()
	} else if r.SearchTask != "" {
		fmt.Fprintf(w, "\nOnline search for %s returned nothing usable (network problem, or every hit was a variant — try --variants).\n", r.SearchTask)
	}
	if r.ListNote != "" {
		fmt.Fprintf(w, "\n%s\n", r.ListNote)
	}
}

func commasInt(n int64) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// Build assembles a ModelResult from a model and the fit table. cal may be nil.
func Build(m *model.Model, aliases []string, extras []string, budgetGB float64, opts fit.Options, bandwidth float64, cal Calibration) ModelResult {
	if opts.KV == "" {
		opts.KV = fit.KVF16
	}
	r := ModelResult{
		Name: m.Name, Aliases: aliases, Path: m.Path, Arch: m.Arch, Quant: m.Quant,
		Params:         Value{float64(m.Params), m.ParamsSource},
		ActiveParams:   Value{float64(m.ActiveParams()), model.Inferred},
		WeightsGB:      Value{float64(m.WeightsBytes) / fit.GiB, model.Measured},
		AttentionLayer: m.AttentionLayers(), Layers: m.Layers, MoE: m.IsMoE(),
		Baseline: m.Baseline, BaselineNote: m.BaselineNote, Tasks: m.Tasks, Remote: m.Remote, Warnings: m.Warnings, Extras: extras,
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
	var fc fit.Calibration
	if cal != nil {
		fc = cal
		if s, ok := cal.Lookup(append([]string{m.Name}, aliases...)...); ok {
			r.MeasuredTokS = &Measured{Value: s.TokPerSec, Source: model.Measured, Backend: s.Backend,
				Context: s.Context, MeasuredAt: s.MeasuredAt.Format("2006-01-02")}
		}
	}
	if s, ok := fit.Estimate(m, bandwidth, fc); ok {
		r.DecodeTokS = &SpeedValue{s.TokPerSec, model.Inferred, s.Confidence, s.Basis}
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
		if len(m.Tasks) > 0 {
			fmt.Fprintf(w, "  for: %s (from the file's own metadata)\n", strings.Join(m.Tasks, ", "))
		}
		tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  context\tKV GB\ttotal GB\tfits")
		for _, row := range m.Rows {
			fmt.Fprintf(tw, "  %s\t%.1f\t%.1f\t%s\t%s\n", ctxLabel(row.Context), row.KVGB, row.TotalGB, row.Verdict, row.Note)
		}
		tw.Flush()
		for _, row := range m.Rows {
			if row.Verdict == fit.Tight {
				fmt.Fprintf(w, "  %s\n", fit.TightNote)
				break
			}
		}
		if m.MaxContext > 0 {
			fmt.Fprintf(w, "  max context that fits: %s (KV per token %.0f bytes, inferred from header)\n", ctxLabel(m.MaxContext), m.KVBytesPerTok.Value)
		} else {
			fmt.Fprintf(w, "  does not fit at any listed context\n")
		}
		switch {
		case m.MeasuredTokS != nil:
			// A measurement of this very model beats any estimate; do not
			// print both and invite the reader to compare them.
			fmt.Fprintf(w, "  decode: %.1f tok/s measured (%s, %s context, %s)\n", m.MeasuredTokS.Value, m.MeasuredTokS.Backend, ctxLabel(uint64(m.MeasuredTokS.Context)), m.MeasuredTokS.MeasuredAt)
		case m.DecodeTokS != nil:
			fmt.Fprintf(w, "  decode estimate: ~%.0f tok/s (inferred, %s confidence; %s)\n", m.DecodeTokS.Value, m.DecodeTokS.Confidence, m.DecodeTokS.Basis)
		}
		for _, x := range m.Extras {
			fmt.Fprintf(w, "  companion: %s (vision projector; needed only for image input)\n", x)
		}
		for _, wmsg := range m.Warnings {
			fmt.Fprintf(w, "  warning: %s\n", wmsg)
		}
	}
	WriteSuggestions(w, r)
}

// WriteCompactTable renders one row per model: used for --all-quants (one
// row per quant of a repo) and for --suggest (installed and suggested
// models side by side), where the full per-model block is too much.
func WriteCompactTable(w io.Writer, r Report, title string) {
	fmt.Fprintf(w, "Machine   %s, %.0f GB, budget %.1f GB for weights + KV\n", r.Hardware.Chip, r.Hardware.RAMGB, r.Budget.GB)
	fmt.Fprintf(w, "%s (weights measured; fits inferred; tok/s measured where marked, else estimated)\n\n", title)
	tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
	var ctxs []uint64
	for _, m := range r.Models {
		if len(m.Rows) > 0 {
			for _, row := range m.Rows {
				ctxs = append(ctxs, row.Context)
			}
			break
		}
	}
	fmt.Fprint(tw, "model\tquant\tGB")
	for _, c := range ctxs {
		fmt.Fprintf(tw, "\t%s", ctxLabel(c))
	}
	fmt.Fprintln(tw, "\ttok/s\tfor\tstatus")
	for _, m := range r.Models {
		if m.Error != "" {
			// Same cell count as a data row, or tabwriter restarts its
			// column alignment at this line.
			fmt.Fprintf(tw, "%s\t-\t-%s\t-\t-\terror: %s\n", m.Name, strings.Repeat("\t-", len(ctxs)), m.Error)
			continue
		}
		fmt.Fprintf(tw, "%s\t%s\t%.1f", m.Name, m.Quant, m.WeightsGB.Value)
		for _, row := range m.Rows {
			fmt.Fprintf(tw, "\t%s", row.Verdict)
		}
		switch {
		case m.MeasuredTokS != nil:
			fmt.Fprintf(tw, "\t%.0f meas", m.MeasuredTokS.Value)
		case m.DecodeTokS != nil:
			fmt.Fprintf(tw, "\t~%.0f", m.DecodeTokS.Value)
		default:
			fmt.Fprint(tw, "\t-")
		}
		fmt.Fprintf(tw, "\t%s", strings.Join(m.Tasks, ","))
		status := "installed"
		switch {
		case m.InstalledAs != "":
			status = "installed as " + m.InstalledAs
		case m.Remote:
			status = fmt.Sprintf("pull %.1f GB", m.WeightsGB.Value)
		}
		if !m.Baseline {
			status += ", variant"
		}
		fmt.Fprintf(tw, "\t%s\n", status)
	}
	tw.Flush()
	fmt.Fprintln(w, "\n"+fit.TightNote)
}

func ctxLabel(n uint64) string {
	if n%1024 == 0 {
		return fmt.Sprintf("%dk", n/1024)
	}
	return fmt.Sprintf("%d", n)
}
