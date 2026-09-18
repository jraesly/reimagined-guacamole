package report

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

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
	res := Build(m, []string{"qwen3.8:27b-32k"}, []string{"/x/mmproj-F32.gguf"}, budget.GB, fit.Options{Contexts: []uint64{32768, 1048576}, KV: fit.KVF16}, info.BandwidthGBs)
	return Report{Hardware: info, Budget: budget, KV: fit.KVF16, Models: []ModelResult{res}}
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
