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
	// qwen35 without an explicit interval defaults to full attention every
	// 4th layer (llama.cpp's fallback): 66/4 = 16 layers hold KV.
	if m.Layers != 66 || m.AttentionLayers() != 16 || m.KeyLen != 128 || m.ValLen != 128 {
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
		{Name: "token_embd.weight", Dims: []uint64{1000}, Type: 0},              // 4000 B: row lookup, not read (output.weight exists)
		{Name: "output.weight", Dims: []uint64{100}, Type: 1},                   // 200 B always
		{Name: "blk.0.attn_q.weight", Dims: []uint64{1000}, Type: 1},            // 2000 B always
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 100}, Type: 0}, // 25600 routed elems, 102400 B
		{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{400}, Type: 1},     // 800 B shared: always active
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
	if m.TotalElems != 28100 || m.ExpertElems != 25600 {
		t.Errorf("elems total=%d expert=%d", m.TotalElems, m.ExpertElems)
	}
	// always-active 2500 + 25600 * 8/256 = 3300 of 28100
	if want := 3300.0 / 28100; math.Abs(m.ActiveFraction()-want) > 1e-9 {
		t.Errorf("active fraction = %v, want %v", m.ActiveFraction(), want)
	}
	if m.ActiveParams() != 3300 {
		t.Errorf("active params = %d", m.ActiveParams())
	}
	// bytes read per token: 200 + 2000 + 800 always + 102400 * 8/256 = 6200;
	// the embedding table is a row lookup and does not count.
	if b, src := m.BytesPerToken(); b != 6200 || src != Inferred {
		t.Errorf("bytes/token = %v (%s), want 6200 inferred", b, src)
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

func TestFromGGUFMissingKVHeadsAndHeadCountIsError(t *testing.T) {
	f := dense()
	delete(f.Metadata, "qwen35.attention.head_count_kv")
	delete(f.Metadata, "qwen35.attention.head_count")
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	if _, err := FromGGUF(p); err == nil || !strings.Contains(err.Error(), "head_count") {
		t.Errorf("err = %v", err)
	}
}

func TestHybridPrecedenceAndDefaults(t *testing.T) {
	base := func() gguf.Fixture {
		f := dense()
		f.Metadata["qwen35.block_count"] = uint32(9)
		f.Metadata["qwen35.nextn_predict_layers"] = uint32(1)
		return f
	}
	p := filepath.Join(t.TempDir(), "m.gguf")

	// No interval key: the qwen35 family defaults to every 4th layer → 3,7 + nextn 8.
	writeFixture(t, p, base(), 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 3 {
		t.Errorf("default interval: attention layers = %d, want 3", m.AttentionLayers())
	}

	// Explicit recurrent_layers mask wins over the interval.
	f := base()
	f.Metadata["qwen35.full_attention_interval"] = uint32(4)
	f.Metadata["qwen35.attention.recurrent_layers"] = []uint32{1, 1, 1, 1, 1, 1, 1, 0, 0}
	writeFixture(t, p, f, 0)
	m, err = FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 2 || m.KVHeads[7] == 0 || m.KVHeads[8] == 0 {
		t.Errorf("mask: attention layers = %d kv=%v", m.AttentionLayers(), m.KVHeads)
	}

	// Index-list form of the mask.
	f = base()
	f.Metadata["qwen35.attention.recurrent_layers"] = []uint32{0, 1, 2}
	writeFixture(t, p, f, 0)
	m, _ = FromGGUF(p)
	if m.AttentionLayers() != 6 {
		t.Errorf("index mask: attention layers = %d, want 6", m.AttentionLayers())
	}

	// A non-hybrid architecture without the key keeps every layer.
	f = base()
	f.Metadata["general.architecture"] = "llama"
	for k, v := range map[string]any{"llama.block_count": uint32(9), "llama.attention.head_count": uint32(40),
		"llama.attention.head_count_kv": uint32(8), "llama.attention.key_length": uint32(128)} {
		f.Metadata[k] = v
	}
	writeFixture(t, p, f, 0)
	m, err = FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttentionLayers() != 9 {
		t.Errorf("llama: attention layers = %d, want 9", m.AttentionLayers())
	}

	// nextn larger than block_count is a corrupt header, not a panic.
	f = base()
	f.Metadata["qwen35.nextn_predict_layers"] = uint32(50)
	writeFixture(t, p, f, 0)
	if _, err := FromGGUF(p); err == nil || !strings.Contains(err.Error(), "nextn") {
		t.Errorf("nextn overflow: %v", err)
	}
}

func TestMissingKVHeadsDefaultsToMHA(t *testing.T) {
	f := dense()
	f.Metadata["general.architecture"] = "llama"
	for k, v := range map[string]any{"llama.block_count": uint32(2), "llama.attention.head_count": uint32(32),
		"llama.attention.key_length": uint32(128)} {
		f.Metadata[k] = v
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.KVHeads[0] != 32 || m.KVHeads[1] != 32 {
		t.Errorf("kv heads = %v, want head_count (32) on every layer", m.KVHeads)
	}
}

func TestCacheModelWarningsContextAndSplit(t *testing.T) {
	f := dense()
	f.Metadata["qwen35.attention.kv_lora_rank"] = uint32(512)
	f.Metadata["qwen35.context_length"] = uint32(262144)
	f.Metadata["split.count"] = uint16(3)
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.CacheModel != "mla" || m.ContextLength != 262144 {
		t.Errorf("cache=%q ctx=%d", m.CacheModel, m.ContextLength)
	}
	joined := strings.Join(m.Warnings, "\n")
	if !strings.Contains(joined, "upper bound") || !strings.Contains(joined, "declares 3 shards") {
		t.Errorf("warnings = %v", m.Warnings)
	}
	f = dense()
	f.Metadata["qwen35.attention.sliding_window"] = uint32(4096)
	writeFixture(t, p, f, 0)
	if m, _ := FromGGUF(p); m.CacheModel != "swa" {
		t.Errorf("swa cache model = %q", m.CacheModel)
	}
}

func TestTiedEmbeddingsCountAsOutputHead(t *testing.T) {
	f := dense()
	f.Tensors = []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{1000}, Type: 1},   // 2000 B
		{Name: "blk.0.attn_q.weight", Dims: []uint64{256}, Type: 12}, // 144 B
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if b, src := m.BytesPerToken(); b != 2144 || src != Inferred {
		t.Errorf("tied embeddings: bytes/token = %v (%s), want 2144 inferred", b, src)
	}
	f.Tensors = append(f.Tensors, gguf.TensorInfo{Name: "output.weight", Dims: []uint64{64}, Type: 8}) // 68 B
	writeFixture(t, p, f, 0)
	m, _ = FromGGUF(p)
	if b, _ := m.BytesPerToken(); b != 212 {
		t.Errorf("untied: bytes/token = %v, want 212", b)
	}
}

func TestPartialHeaderNeverFeedsExpertFraction(t *testing.T) {
	f := dense()
	f.Metadata["general.architecture"] = "qwen35moe"
	for k, v := range map[string]any{
		"qwen35moe.block_count": uint32(4), "qwen35moe.attention.head_count": uint32(16),
		"qwen35moe.attention.head_count_kv": uint32(2), "qwen35moe.attention.key_length": uint32(128),
		"qwen35moe.expert_count": uint32(256), "qwen35moe.expert_used_count": uint32(8),
	} {
		f.Metadata[k] = v
	}
	f.Tensors = []gguf.TensorInfo{{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 100}, Type: 0}}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	h, err := gguf.ReadHeader(p)
	if err != nil {
		t.Fatal(err)
	}
	m, err := FromHeader(h, "hf:x/y", "y", 10_000_000_000, "x", true)
	if err != nil {
		t.Fatal(err)
	}
	if m.TotalElems != 0 || m.ExpertElems != 0 || m.ReadBytes != 0 {
		t.Errorf("partial header leaked tensor sums: total=%d expert=%d read=%d", m.TotalElems, m.ExpertElems, m.ReadBytes)
	}
	if m.ActiveFraction() != 8.0/256 {
		t.Errorf("active fraction should fall back to used/total, got %v", m.ActiveFraction())
	}
}

func TestDenseReadBytesAndUnknownType(t *testing.T) {
	f := dense()
	f.Tensors = []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{1000}, Type: 1},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{256}, Type: 12}, // one Q4_K block = 144 B
		{Name: "output.weight", Dims: []uint64{64}, Type: 8},         // two Q8_0 blocks = 68 B
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeFixture(t, p, f, 0)
	m, err := FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if b, src := m.BytesPerToken(); b != 212 || src != Inferred {
		t.Errorf("dense bytes/token = %v (%s), want 212 inferred", b, src)
	}

	f.Tensors = append(f.Tensors, gguf.TensorInfo{Name: "blk.1.weird", Dims: []uint64{8}, Type: 4})
	writeFixture(t, p, f, 0)
	m, err = FromGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.ReadBytes != 0 {
		t.Errorf("unknown tensor type must disable tensor-level reads, got %d", m.ReadBytes)
	}
	if b, src := m.BytesPerToken(); b != float64(m.WeightsBytes) || src != Inferred {
		t.Errorf("fallback bytes/token = %v (%s)", b, src)
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
