package model

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
)

func writeFixture(t *testing.T, path string, f gguf.Fixture, padding int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	data := append(f.Bytes(), make([]byte, padding)...)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func dense() gguf.Fixture {
	return gguf.Fixture{Metadata: map[string]any{
		"general.architecture":           "qwen35",
		"general.name":                   "Qwen3.8 27B",
		"general.parameter_count":        uint64(27_300_000_000),
		"general.file_type":              uint32(15),
		"qwen35.block_count":             uint32(66),
		"qwen35.attention.head_count":    uint32(40),
		"qwen35.attention.head_count_kv": uint32(8),
		"qwen35.embedding_length":        uint32(5120),
		"qwen35.attention.key_length":    uint32(128),
	}}
}

func TestFromGGUFDense(t *testing.T) {
	p := filepath.Join(t.TempDir(), "unsloth", "Qwen3.8-27B-GGUF", "q4.gguf")
	writeFixture(t, p, dense(), 1000)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.Params != 27_300_000_000 || m.ParamsSource != Measured {
		t.Errorf("params = %d (%s)", m.Params, m.ParamsSource)
	}
	if m.Quant != "Q4_K_M" {
		t.Errorf("quant = %s", m.Quant)
	}
	if m.Layers != 66 || m.AttentionLayers() != 66 || m.KeyLen != 128 || m.ValLen != 128 {
		t.Errorf("geometry = layers %d attn %d k %d v %d", m.Layers, m.AttentionLayers(), m.KeyLen, m.ValLen)
	}
	if m.IsMoE() || m.ActiveFraction() != 1 {
		t.Error("dense model reported as MoE")
	}
	if m.Owner != "unsloth" || !m.Baseline {
		t.Errorf("owner=%q baseline=%v note=%q", m.Owner, m.Baseline, m.BaselineNote)
	}
	if m.WeightsBytes < 1000 {
		t.Errorf("weights bytes = %d", m.WeightsBytes)
	}
}

func TestFromGGUFHybridPerLayerKV(t *testing.T) {
	f := dense()
	f.Metadata["qwen35.block_count"] = uint32(8)
	f.Metadata["qwen35.attention.head_count_kv"] = []uint32{0, 0, 0, 8, 0, 0, 0, 8}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 2 {
		t.Errorf("attention layers = %d, want 2", m.AttentionLayers())
	}
}

// Values below are the real qwen35 / qwen35moe headers read from Ollama's
// blobs on 2026-09-17, and the expected KV sizes are what Ollama's server log
// allocated at 262144 context: 16384+1024 MiB (27B) and 5120+512 MiB (35B).
func TestFromGGUFHybridFullAttentionInterval(t *testing.T) {
	f := dense()
	for k, v := range map[string]any{
		"qwen35.block_count": uint32(65), "qwen35.attention.head_count": uint32(24),
		"qwen35.attention.head_count_kv": uint32(4), "qwen35.attention.key_length": uint32(256),
		"qwen35.attention.value_length": uint32(256), "qwen35.full_attention_interval": uint32(4),
		"qwen35.nextn_predict_layers": uint32(1),
	} {
		f.Metadata[k] = v
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 17 {
		t.Fatalf("attention layers = %d, want 16 full-attention + 1 nextn", m.AttentionLayers())
	}
	// layer indexes 3,7,...,63 plus the nextn layer 64
	for i, h := range m.KVHeads {
		wantKV := (i+1)%4 == 0 || i == 64
		if (h > 0) != wantKV {
			t.Errorf("layer %d kv heads = %d", i, h)
		}
	}
	perTok := 0.0
	for _, h := range m.KVHeads {
		perTok += float64(h) * float64(m.KeyLen+m.ValLen) * 2
	}
	if perTok != 69632 || perTok*262144/(1024*1024) != 16384+1024 {
		t.Errorf("kv bytes/token = %v", perTok)
	}
}

func TestFromGGUFMoEExpertAccounting(t *testing.T) {
	f := dense()
	f.Metadata["general.architecture"] = "qwen35moe"
	delete(f.Metadata, "general.parameter_count")
	for k, v := range map[string]any{
		"qwen35moe.block_count": uint32(41), "qwen35moe.attention.head_count": uint32(16),
		"qwen35moe.attention.head_count_kv": uint32(2), "qwen35moe.embedding_length": uint32(2048),
		"qwen35moe.attention.key_length": uint32(256), "qwen35moe.attention.value_length": uint32(256),
		"qwen35moe.expert_count": uint32(256), "qwen35moe.expert_used_count": uint32(8),
		"qwen35moe.full_attention_interval": uint32(4), "qwen35moe.nextn_predict_layers": uint32(1),
	} {
		f.Metadata[k] = v
	}
	f.Tensors = []gguf.TensorInfo{
		{Name: "blk.0.attn_q.weight", Dims: []uint64{1000}},
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 100}}, // 25600 routed
		{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{400}},     // shared: always active
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 11 {
		t.Errorf("attention layers = %d, want 10 + 1 nextn", m.AttentionLayers())
	}
	if m.TotalElems != 27000 || m.ExpertElems != 25600 {
		t.Errorf("elems total=%d expert=%d", m.TotalElems, m.ExpertElems)
	}
	// always-active 1400 + 25600 * 8/256 = 2200 of 27000
	if want := 2200.0 / 27000; math.Abs(m.ActiveFraction()-want) > 1e-9 {
		t.Errorf("active fraction = %v, want %v", m.ActiveFraction(), want)
	}
	if m.ActiveParams() != 2200 {
		t.Errorf("active params = %d", m.ActiveParams())
	}
}

func TestFromGGUFMoE(t *testing.T) {
	f := dense()
	f.Metadata["general.architecture"] = "qwen35moe"
	for k, v := range map[string]any{
		"qwen35moe.block_count": uint32(40), "qwen35moe.attention.head_count": uint32(16),
		"qwen35moe.attention.head_count_kv": uint32(2), "qwen35moe.embedding_length": uint32(2048),
		"qwen35moe.attention.key_length": uint32(128), "qwen35moe.expert_count": uint32(128),
		"qwen35moe.expert_used_count": uint32(8),
	} {
		f.Metadata[k] = v
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsMoE() || m.ActiveFraction() != 8.0/128 {
		t.Errorf("moe: %v %v", m.IsMoE(), m.ActiveFraction())
	}
}

func TestFromGGUFMissingKeyLengthFallsBack(t *testing.T) {
	f := dense()
	delete(f.Metadata, "qwen35.attention.key_length")
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.KeyLen != 5120/40 {
		t.Errorf("key len = %d", m.KeyLen)
	}
	if len(m.Warnings) == 0 || !strings.Contains(m.Warnings[0], "key_length") {
		t.Errorf("expected warning, got %v", m.Warnings)
	}
}

func TestFromGGUFMissingKVHeadsIsError(t *testing.T) {
	f := dense()
	delete(f.Metadata, "qwen35.attention.head_count_kv")
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	if _, err := FromGGUF(p); err == nil || !strings.Contains(err.Error(), "head_count_kv") {
		t.Errorf("err = %v", err)
	}
}

func TestFromGGUFSumsTensorsWhenParamCountMissing(t *testing.T) {
	f := dense()
	delete(f.Metadata, "general.parameter_count")
	f.Tensors = []gguf.TensorInfo{
		{Name: "a", Dims: []uint64{10, 20}},
		{Name: "b", Dims: []uint64{5}},
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.Params != 205 || m.ParamsSource != Observed {
		t.Errorf("params = %d (%s)", m.Params, m.ParamsSource)
	}
}

func TestShardsResolvedFromAnyShard(t *testing.T) {
	dir := t.TempDir()
	f := dense()
	for i := 1; i <= 3; i++ {
		writeFixture(t, filepath.Join(dir, "m-0000"+string(rune('0'+i))+"-of-00003.gguf"), f, 100*i)
	}
	m, err := FromGGUF(filepath.Join(dir, "m-00002-of-00003.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Files) != 3 || filepath.Base(m.Path) != "m-00001-of-00003.gguf" {
		t.Errorf("files = %v path = %s", m.Files, m.Path)
	}
	header := uint64(len(f.Bytes()))
	if want := 3*header + 600; m.WeightsBytes != want {
		t.Errorf("weights = %d, want %d", m.WeightsBytes, want)
	}
}

func TestShardsMissingIsError(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, filepath.Join(dir, "m-00001-of-00002.gguf"), dense(), 0)
	if _, err := FromGGUF(filepath.Join(dir, "m-00001-of-00002.gguf")); err == nil || !strings.Contains(err.Error(), "missing shard") {
		t.Errorf("err = %v", err)
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		owner, name string
		baseline    bool
	}{
		{"unsloth", "Qwen3.8-27B", true},
		{"Qwen", "Qwen3.8-27B", true},
		{"DavidAU", "Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-Uncensored", false},
		{"unsloth", "Qwen3.8-27B-Uncensored", false},
		{"", "Qwen3.8-27B", false},
		{"someone", "Qwen3.8-27B", false},
	}
	for _, c := range cases {
		got, note := Classify(c.owner, c.name)
		if got != c.baseline {
			t.Errorf("Classify(%q,%q) = %v (%s), want %v", c.owner, c.name, got, note, c.baseline)
		}
	}
}

func TestFromMLXDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "mlx-community", "Qwen3.8-27B-4bit")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := `{"model_type":"qwen3","num_hidden_layers":64,"num_key_value_heads":8,"num_attention_heads":64,"hidden_size":8192,"quantization":{"bits":4,"group_size":64}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), make([]byte, 4500), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := FromMLXDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if m.Layers != 64 || m.KeyLen != 128 || m.Quant != "4-bit" || m.WeightsBytes != 4500 {
		t.Errorf("model = %+v", m)
	}
	if m.Params != uint64(4500*8/4.5) || m.ParamsSource != Inferred {
		t.Errorf("params = %d (%s)", m.Params, m.ParamsSource)
	}
	if m.Baseline {
		t.Errorf("mlx-community should not be classified as baseline: %s", m.BaselineNote)
	}
}

func TestFromMLXDirNoWeights(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"num_hidden_layers":2,"num_key_value_heads":1,"head_dim":64}`), 0o644)
	if _, err := FromMLXDir(dir); err == nil || !strings.Contains(err.Error(), "safetensors") {
		t.Errorf("err = %v", err)
	}
}
