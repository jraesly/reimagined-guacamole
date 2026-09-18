package report

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/calib"
	"github.com/jraesly/reimagined-guacamole/internal/fit"
	"github.com/jraesly/reimagined-guacamole/internal/hw"
	"github.com/jraesly/reimagined-guacamole/internal/model"
)

func sample() Report {
	kv := make([]uint32, 8)
	kv[3], kv[7] = 4, 4
	m := &model.Model{
		Name: "qwen3.8:27b", Path: "/blobs/x", Arch: "qwen35", Quant: "Q4_K_M",
		Params: 27_300_000_000, ParamsSource: model.Observed, Layers: 8, KVHeads: kv,
		KeyLen: 256, ValLen: 256, WeightsBytes: 16_900_000_000, Baseline: true, BaselineNote: "baseline (library)",
		Warnings: []string{"general.parameter_count missing; summed tensor elements instead"},
	}
	info, _ := hw.ParseDarwin("Apple M1 Max", "34359738368", "0")
	budget := info.Budget(8)
	// 2 KV layers × 4 heads × 512 × 2 B = 8 KiB/token: 0.25 GB at 32k (fits),
	// 8 GB at 1M (15.7 + 8 + 0.5 > 24 GB budget: no).
	res := Build(m, []string{"qwen3.8:27b-32k"}, []string{"/x/mmproj-F32.gguf"}, budget.GB, fit.Options{Contexts: []uint64{32768, 1048576}, KV: fit.KVF16}, info.BandwidthGBs, nil)
	return Report{Hardware: info, Budget: budget, KV: fit.KVF16, Models: []ModelResult{res}}
}

func TestWriteQuantTable(t *testing.T) {
	r := sample()
	q8 := r.Models[0]
	q8.Name, q8.Quant = "Q-Q8_0.gguf", "Q8_0"
	broken := ModelResult{Name: "Q-broken.gguf", Error: "not a GGUF file"}
	r.Models = []ModelResult{r.Models[0], q8, broken}
	r.Models[0].Name, r.Models[0].Quant = "Q-Q4_K_M.gguf", "Q4_K_M"
	var b bytes.Buffer
	WriteQuantTable(&b, r, "unsloth/Q")
	out := b.String()
	for _, want := range []string{"file", "quant", "32k", "1024k", "est tok/s", "Q-Q4_K_M.gguf", "Q4_K_M", "Q-Q8_0.gguf", "yes", "no", "~11", "error: not a GGUF file", "tight = within 10%"} {
		if !strings.Contains(out, want) {
			t.Errorf("quant table missing %q:\n%s", want, out)
		}
	}
	if strings.Count(out, "\n") > 12 {
		t.Errorf("table should be compact, got %d lines:\n%s", strings.Count(out, "\n"), out)
	}
}

func TestBuildUsesCalibration(t *testing.T) {
	cf := &calib.File{Version: 1}
	cf.Add(calib.Sample{Model: "qwen3.8:27b-32k", Kind: "dense", Backend: "ollama", TokPerSec: 11.35,
		BytesPerToken: 16e9, Context: 4096, MeasuredAt: time.Date(2026, 9, 17, 0, 0, 0, 0, time.UTC)})
	r := sample()
	m := &model.Model{Name: "qwen3.8:27b", Params: 1, Layers: 1, KVHeads: []uint32{1}, KeyLen: 1, ValLen: 1, WeightsBytes: 8e9}
	res := Build(m, []string{"qwen3.8:27b-32k"}, nil, r.Budget.GB, fit.Options{}, 0, cf)
	if res.MeasuredTokS == nil || res.MeasuredTokS.Value != 11.35 || res.MeasuredTokS.MeasuredAt != "2026-09-17" {
		t.Errorf("measured = %+v (alias lookup should find the sample)", res.MeasuredTokS)
	}
	// 16e9 × 11.35 = 181.6 GB/s effective; 8 GB/token → 22.7 tok/s, calibrated basis, no bandwidth needed.
	if res.DecodeTokS == nil || res.DecodeTokS.Value < 22 || res.DecodeTokS.Value > 23.5 || !strings.Contains(res.DecodeTokS.Basis, "calibrated") {
		t.Errorf("estimate = %+v", res.DecodeTokS)
	}
	var b bytes.Buffer
	WriteText(&b, Report{Hardware: r.Hardware, Budget: r.Budget, KV: fit.KVF16, Models: []ModelResult{res}})
	if !strings.Contains(b.String(), "decode: 11.3 tok/s measured (ollama, 4k context, 2026-09-17)") {
		t.Errorf("text missing measured line:\n%s", b.String())
	}
}

func TestWriteTextContainsKeyFacts(t *testing.T) {
	var b bytes.Buffer
	WriteText(&b, sample())
	out := b.String()
	for _, want := range []string{
		"Apple M1 Max, 32 GB unified, 400 GB/s",
		"Budget    24.0 GB",
		"qwen3.8:27b  (qwen3.8:27b-32k)",
		"2/8 layers hold KV",
		"baseline (library)",
		"32k", "1024k", "yes", "no",
		"max context that fits: 32k",
		"decode estimate: ~11 tok/s (inferred, medium confidence",
		"reclaimable: /x/mmproj-F32.gguf",
		"warning: general.parameter_count missing",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("text report missing %q\n%s", want, out)
		}
	}
}

func TestWriteTextReportsErrors(t *testing.T) {
	var b bytes.Buffer
	WriteText(&b, Report{Models: []ModelResult{{Name: "broken.gguf", Error: "not a GGUF file"}}})
	if !strings.Contains(b.String(), "error: not a GGUF file") {
		t.Errorf("error not rendered:\n%s", b.String())
	}
}

func TestWriteJSONLabelsEveryNumber(t *testing.T) {
	var b bytes.Buffer
	if err := WriteJSON(&b, sample()); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(b.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	m := got.Models[0]
	for _, key := range []string{"params", "active_params", "weights_gb", "kv_bytes_per_token", "decode_tok_s"} {
		v, ok := m[key].(map[string]any)
		if !ok {
			t.Errorf("%s is not an object: %v", key, m[key])
			continue
		}
		if src, _ := v["source"].(string); src != "measured" && src != "observed" && src != "inferred" {
			t.Errorf("%s has bad source %q", key, src)
		}
	}
	if m["weights_gb"].(map[string]any)["source"] != "measured" || m["kv_bytes_per_token"].(map[string]any)["source"] != "inferred" {
		t.Errorf("wrong provenance: weights=%v kv=%v", m["weights_gb"], m["kv_bytes_per_token"])
	}
	if rows, _ := m["rows"].([]any); len(rows) != 2 {
		t.Errorf("rows = %v", m["rows"])
	}
}
