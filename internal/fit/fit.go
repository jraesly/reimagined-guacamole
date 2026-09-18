// Package fit answers "does this model fit on this machine, at what context,
// and roughly how fast" from arithmetic on the model header and hardware.
package fit

import (
	"fmt"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

const GiB = 1024 * 1024 * 1024

// KVType is the KV-cache element type the server will use.
type KVType string

const (
	KVF16  KVType = "f16"
	KVQ8_0 KVType = "q8_0"
	KVQ4_0 KVType = "q4_0"
)

// BytesPerElement is the storage cost of one KV element. Block quants carry a
// per-block scale: q8_0 is 34 bytes per 32 elements, q4_0 is 18 per 32.
func (k KVType) BytesPerElement() (float64, error) {
	switch k {
	case KVF16:
		return 2, nil
	case KVQ8_0:
		return 34.0 / 32, nil
	case KVQ4_0:
		return 18.0 / 32, nil
	}
	return 0, fmt.Errorf("unknown kv type %q (want f16, q8_0 or q4_0)", k)
}

// KVBytesPerToken is the cache growth per token across all attention layers.
func KVBytesPerToken(m *model.Model, kv KVType) (float64, error) {
	bpe, err := kv.BytesPerElement()
	if err != nil {
		return 0, err
	}
	total := 0.0
	for _, heads := range m.KVHeads {
		total += float64(heads) * float64(m.KeyLen+m.ValLen) * bpe
	}
	return total, nil
}

// Verdict is the fit classification for one context size.
type Verdict string

const (
	Yes   Verdict = "yes"
	Tight Verdict = "tight" // within 10% of the budget: runtime buffers are not modeled, so it may still spill
	No    Verdict = "no"
)

// TightNote is printed wherever a tight verdict appears.
const TightNote = "tight = within 10% of the budget; runtime buffers are not modeled, so it may still spill (validated once: an 11% CPU spill at a tight verdict)"

// Row is one line of the fit table.
type Row struct {
	Context  uint64
	KVGB     float64
	TotalGB  float64
	BudgetGB float64
	Verdict  Verdict
	Note     string `json:",omitempty"` // e.g. the context exceeds the model's maximum
}

// Options tune the table.
type Options struct {
	Contexts  []uint64
	KV        KVType
	ComputeGB float64 // scratch buffers the runtime allocates beyond weights and KV
}

// DefaultContexts are the sizes shown when the user does not pick one.
var DefaultContexts = []uint64{8192, 16384, 32768, 65536, 131072}

// Classify compares a total against a budget.
func Classify(totalGB, budgetGB float64) Verdict {
	switch {
	case totalGB <= budgetGB*0.9:
		return Yes
	case totalGB <= budgetGB:
		return Tight
	}
	return No
}

// Table computes fit rows for each context in opts.Contexts.
func Table(m *model.Model, budgetGB float64, opts Options) ([]Row, error) {
	if opts.KV == "" {
		opts.KV = KVF16
	}
	if len(opts.Contexts) == 0 {
		opts.Contexts = DefaultContexts
	}
	if opts.ComputeGB == 0 {
		// Ollama's own log on the calibration machine showed ~0.8 GB of
		// compute buffers plus ~0.2 GB of recurrent state for a hybrid MoE;
		// 0.5 GB let a "tight" verdict spill 11% to CPU in validation.
		opts.ComputeGB = 1.0
	}
	perTok, err := KVBytesPerToken(m, opts.KV)
	if err != nil {
		return nil, err
	}
	weightsGB := float64(m.WeightsBytes) / GiB
	rows := make([]Row, 0, len(opts.Contexts))
	for _, ctx := range opts.Contexts {
		kvGB := perTok * float64(ctx) / GiB
		total := weightsGB + kvGB + opts.ComputeGB
		row := Row{Context: ctx, KVGB: kvGB, TotalGB: total, BudgetGB: budgetGB, Verdict: Classify(total, budgetGB)}
		if m.ContextLength > 0 && ctx > m.ContextLength {
			row.Note = fmt.Sprintf("exceeds the model's %dk context", m.ContextLength/1024)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// MaxContext returns the largest context in rows that fits (yes or tight).
func MaxContext(rows []Row) (uint64, bool) {
	best := uint64(0)
	for _, r := range rows {
		if r.Verdict != No && r.Context > best {
			best = r.Context
		}
	}
	return best, best > 0
}

// Efficiency factors convert the memory-bandwidth roofline into a realistic
// decode rate. Calibrated on an Apple M1 Max (400 GB/s) with Ollama 0.34,
// 2026-09-16, `ollama run --verbose`, 32k context:
//   - dense: Qwen3.8-27B Q4_K_M, 16.9 GB weights → measured 11.35 tok/s;
//     roofline 400/16.9 = 23.7 → 0.48
//   - moe:   Qwen3.6-35B-A3B Q4_K_M, 21.7 GB weights, ~3.5B of 35.5B params
//     active (routed-expert tensors × 8/256 + everything else) → 2.14 GB
//     read per token; roofline 400/2.14 = 187 → measured 23.5 → 0.125
//
// MoE decode is bounded by more than bandwidth (routing, small matmuls,
// per-expert overhead), hence the much lower factor and low confidence.
// Both are heuristics and every result is labeled inferred.
const (
	denseEfficiency = 0.48
	moeEfficiency   = 0.125
)

// Speed is an estimated decode rate.
type Speed struct {
	TokPerSec  float64
	Confidence string // "medium" for dense, "low" for MoE
	Basis      string // how the number was produced
}

// Calibration is what the machine has measured so far; nil means none.
type Calibration interface {
	// Effective returns the median effective bandwidth (GB/s actually
	// achieved) for a model kind and how many samples back it.
	Effective(kind string) (float64, int)
}

// Kind classifies a model for calibration purposes.
func Kind(m *model.Model) string {
	if m.IsMoE() {
		return "moe"
	}
	return "dense"
}

// Estimate picks the best available basis: this machine's measurements of
// the same kind of model, else the chip bandwidth table with a default
// efficiency. Both are inferred; the report says which was used.
func Estimate(m *model.Model, bandwidthGBs float64, cal Calibration) (Speed, bool) {
	bytesPerTok, _ := m.BytesPerToken()
	if bytesPerTok == 0 {
		return Speed{}, false
	}
	if cal != nil {
		if eff, n := cal.Effective(Kind(m)); n > 0 {
			// One run on one model is a data point, not a calibration.
			conf := "medium"
			if n >= 2 && !m.IsMoE() {
				conf = "high"
			}
			return Speed{
				TokPerSec:  eff / (bytesPerTok / 1e9),
				Confidence: conf,
				Basis:      fmt.Sprintf("calibrated from %d measured %s run(s) on this machine", n, Kind(m)),
			}, true
		}
	}
	s, ok := DecodeEstimate(m, bandwidthGBs)
	if !ok {
		return Speed{}, false
	}
	s.Basis = fmt.Sprintf("chip bandwidth table × default %s efficiency; run fit --measure to calibrate", Kind(m))
	return s, true
}

// DecodeEstimate estimates decode tokens/second from memory bandwidth and
// the bytes the model reads per token.
func DecodeEstimate(m *model.Model, bandwidthGBs float64) (Speed, bool) {
	bytesPerTok, _ := m.BytesPerToken()
	if bandwidthGBs <= 0 || bytesPerTok == 0 {
		return Speed{}, false
	}
	roofline := bandwidthGBs / (bytesPerTok / 1e9)
	if m.IsMoE() {
		return Speed{TokPerSec: roofline * moeEfficiency, Confidence: "low"}, true
	}
	return Speed{TokPerSec: roofline * denseEfficiency, Confidence: "medium"}, true
}
