package fit

import (
	"math"
	"strings"
	"testing"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// qwen38 mirrors the real Qwen3.8-27B Q4_K_M header (qwen35, 64 main layers
// with full attention every 4th, 4 KV heads × 256, 16.9 GB weights). The
// trailing next-token-prediction layer is left out so the numbers match the
// main KV cache line in Ollama's log: 16384 MiB at 262144 context and
// 2048 MiB at 32768.
func qwen38() *model.Model {
	kv := make([]uint32, 64)
	for i := 3; i < 64; i += 4 {
		kv[i] = 4
	}
	return &model.Model{
		Name: "qwen3.8:27b", Params: 27_300_000_000, Layers: 64, KVHeads: kv,
		KeyLen: 256, ValLen: 256, WeightsBytes: 16_900_000_000,
	}
}

// qwen36moe mirrors Qwen3.6-35B-A3B Q4_K_M: 40 main layers, 2 KV heads × 256,
// 256 experts with 8 used, 21.7 GB weights, and expert tensors holding 33.0B
// of the 35.5B parameters so ~3.5B are active per token.
func qwen36moe() *model.Model {
	kv := make([]uint32, 40)
	for i := 3; i < 40; i += 4 {
		kv[i] = 2
	}
	return &model.Model{
		Name: "qwen3.6:35b", Params: 35_500_000_000, Layers: 40, KVHeads: kv,
		KeyLen: 256, ValLen: 256, Experts: 256, ExpertsUsed: 8, WeightsBytes: 21_700_000_000,
		TotalElems: 35_500_000_000, ExpertElems: 33_000_000_000,
	}
}

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestKVBytesPerTokenMatchesOllamaLog(t *testing.T) {
	per, err := KVBytesPerToken(qwen38(), KVF16)
	if err != nil {
		t.Fatal(err)
	}
	// 16 layers × 4 heads × (256+256) × 2 bytes = 65536 bytes/token
	if per != 65536 {
		t.Fatalf("per-token = %v, want 65536", per)
	}
	if got := per * 262144 / (1024 * 1024); got != 16384 {
		t.Errorf("KV at 262144 = %v MiB, want 16384", got)
	}
	if got := per * 32768 / (1024 * 1024); got != 2048 {
		t.Errorf("KV at 32768 = %v MiB, want 2048", got)
	}
}

func TestKVQuantTypes(t *testing.T) {
	m := qwen38()
	f16, _ := KVBytesPerToken(m, KVF16)
	q8, _ := KVBytesPerToken(m, KVQ8_0)
	q4, _ := KVBytesPerToken(m, KVQ4_0)
	if !approx(q8/f16, 34.0/64, 1e-9) || !approx(q4/f16, 18.0/64, 1e-9) {
		t.Errorf("ratios q8=%v q4=%v", q8/f16, q4/f16)
	}
	if _, err := KVBytesPerToken(m, KVType("int3")); err == nil {
		t.Error("unknown kv type should error")
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		total, budget float64
		want          Verdict
	}{
		{10, 24, Yes}, {21.6, 24, Yes}, {21.7, 24, Tight}, {24, 24, Tight}, {24.01, 24, No},
	}
	for _, c := range cases {
		if got := Classify(c.total, c.budget); got != c.want {
			t.Errorf("Classify(%v,%v) = %s, want %s", c.total, c.budget, got, c.want)
		}
	}
}

func TestTableM1Max32GB(t *testing.T) {
	// Budget observed on the M1 Max: Metal reported 25559 MiB total; minus
	// the runtime's own ~1 GB leaves about 24 GB for weights + KV.
	rows, err := Table(qwen38(), 24, Options{})
	if err != nil {
		t.Fatal(err)
	}
	want := map[uint64]Verdict{8192: Yes, 16384: Yes, 32768: Yes, 65536: Yes, 131072: No}
	for _, r := range rows {
		if r.Verdict != want[r.Context] {
			t.Errorf("ctx %d: total %.1f GB verdict %s, want %s", r.Context, r.TotalGB, r.Verdict, want[r.Context])
		}
	}
	if got, _ := MaxContext(rows); got != 65536 {
		t.Errorf("max context = %d", got)
	}
	// 256k must be a clear no: 15.7 + 16 + 1.0 GB on a 24 GB budget.
	rows, _ = Table(qwen38(), 24, Options{Contexts: []uint64{262144}})
	if rows[0].Verdict != No || !approx(rows[0].TotalGB, 32.8, 0.2) {
		t.Errorf("262k: %+v", rows[0])
	}
}

func TestTableDefaults(t *testing.T) {
	rows, err := Table(qwen38(), 24, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(DefaultContexts) {
		t.Errorf("rows = %d", len(rows))
	}
	if _, ok := MaxContext([]Row{{Context: 8192, Verdict: No}}); ok {
		t.Error("MaxContext should report false when nothing fits")
	}
}

type fakeCal map[string]struct {
	eff float64
	n   int
}

func (f fakeCal) Effective(kind, backend string) (float64, int) {
	if v, ok := f[kind+"@"+backend]; ok {
		return v.eff, v.n
	}
	v := f[kind]
	return v.eff, v.n
}

func TestEstimatePrefersMachineCalibration(t *testing.T) {
	m := qwen38()
	m.ReadBytes, m.ReadBytesSource = 16_000_000_000, model.Measured
	// No calibration: table basis.
	s, ok := Estimate(m, 400, nil)
	if !ok || !strings.Contains(s.Basis, "table") || s.Confidence != "medium" {
		t.Errorf("table estimate = %+v", s)
	}
	// Calibrated: 180 GB/s effective over 16 GB/token = 11.25 tok/s.
	s, ok = Estimate(m, 400, fakeCal{"dense": {180, 2}})
	if !ok || !approx(s.TokPerSec, 11.25, 0.01) || s.Confidence != "high" || !strings.Contains(s.Basis, "2 measured dense") {
		t.Errorf("calibrated estimate = %+v", s)
	}
	if s, _ := Estimate(m, 400, fakeCal{"dense": {180, 1}}); s.Confidence != "medium" {
		t.Errorf("a single sample must not be high confidence: %+v", s)
	}
	// The model's own host is passed through so same-backend samples win.
	m.Host = "lmstudio"
	if s, _ := Estimate(m, 400, fakeCal{"dense": {180, 3}, "dense@lmstudio": {160, 1}}); !approx(s.TokPerSec, 10, 0.01) || !strings.Contains(s.Basis, "via lmstudio") {
		t.Errorf("host-specific calibration = %+v", s)
	}
	m.Host = ""
	// Context beyond the model's maximum is called out on the row.
	m.ContextLength = 32768
	rows, _ := Table(m, 24, Options{Contexts: []uint64{16384, 65536}})
	if rows[0].Note != "" || !strings.Contains(rows[1].Note, "exceeds the model's 32k context") {
		t.Errorf("notes = %q / %q", rows[0].Note, rows[1].Note)
	}
	// Calibration for the other kind only does not apply.
	s, _ = Estimate(m, 400, fakeCal{"moe": {50, 1}})
	if !strings.Contains(s.Basis, "table") {
		t.Errorf("kind mismatch should fall back to table: %+v", s)
	}
	// Unknown bandwidth and no calibration: no estimate at all.
	if _, ok := Estimate(m, 0, nil); ok {
		t.Error("no basis should yield no estimate")
	}
	// Unknown bandwidth but calibrated: still an estimate.
	if s, ok := Estimate(m, 0, fakeCal{"dense": {180, 1}}); !ok || s.TokPerSec == 0 {
		t.Errorf("calibration should not need the bandwidth table: %+v %v", s, ok)
	}
}

func TestDecodeEstimateCalibration(t *testing.T) {
	// Values measured with `ollama run --verbose` on the M1 Max, 2026-09-16.
	s, ok := DecodeEstimate(qwen38(), 400)
	if !ok || !approx(s.TokPerSec, 11.35, 1.0) || s.Confidence != "medium" {
		t.Errorf("dense estimate = %+v", s)
	}
	m := qwen36moe()
	if ap := m.ActiveParams(); ap < 3_400_000_000 || ap > 3_600_000_000 {
		t.Errorf("active params = %d, want ~3.5B", ap)
	}
	s, ok = DecodeEstimate(m, 400)
	if !ok || !approx(s.TokPerSec, 23.5, 3.0) || s.Confidence != "low" {
		t.Errorf("moe estimate = %+v", s)
	}
	if _, ok := DecodeEstimate(qwen38(), 0); ok {
		t.Error("unknown bandwidth must not produce an estimate")
	}
}
