// Package model turns a model file (local or remote) into the handful of
// facts fit needs: parameter count, layer/KV geometry, quantization, weight
// bytes and whether it is a vendor baseline or a community variant.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
)

// Source says how a number was obtained. Reports must never upgrade a label.
type Source string

const (
	Measured Source = "measured" // read directly from the file or hardware
	Observed Source = "observed" // counted or timed by probe
	Inferred Source = "inferred" // derived by arithmetic or heuristic
)

// Model is the normalized description fit consumes.
type Model struct {
	Name         string
	Path         string // local path, or a registry reference for remote models
	Files        []string
	Remote       bool
	Arch         string
	Quant        string
	Params       uint64
	ParamsSource Source
	Layers       uint32
	KVHeads      []uint32 // one entry per layer; 0 for layers without attention KV
	KeyLen       uint32
	ValLen       uint32
	Experts      uint32
	ExpertsUsed  uint32
	TotalElems   uint64 // sum of tensor elements when the header lists tensors
	ExpertElems  uint64 // elements in routed-expert tensors (not shared experts)
	WeightsBytes uint64
	Owner        string
	Baseline     bool
	BaselineNote string
	Warnings     []string
}

// IsMoE reports whether the model routes tokens to a subset of experts.
func (m *Model) IsMoE() bool { return m.Experts > 1 && m.ExpertsUsed > 0 }

// ActiveFraction is the share of weights read per token. Dense models read
// everything. For MoE models, routed-expert tensors are read used/total of
// the time and everything else (attention, shared experts, embeddings) is
// always read; when tensor sizes are known the split is computed from them,
// otherwise used/total experts is a lower bound.
func (m *Model) ActiveFraction() float64 {
	if !m.IsMoE() {
		return 1
	}
	share := float64(m.ExpertsUsed) / float64(m.Experts)
	if m.TotalElems == 0 || m.ExpertElems == 0 {
		return share
	}
	always := float64(m.TotalElems - m.ExpertElems)
	return (always + float64(m.ExpertElems)*share) / float64(m.TotalElems)
}

// ActiveParams is the parameter count read per token.
func (m *Model) ActiveParams() uint64 {
	return uint64(float64(m.Params) * m.ActiveFraction())
}

// AttentionLayers counts layers that hold a KV cache.
func (m *Model) AttentionLayers() int {
	n := 0
	for _, h := range m.KVHeads {
		if h > 0 {
			n++
		}
	}
	return n
}

// isRoutedExpert reports whether a tensor name belongs to routed experts.
// llama.cpp names them *_exps.weight; shared experts are *_shexp.weight.
func isRoutedExpert(name string) bool {
	return strings.Contains(name, "_exps.")
}

var shardRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// Shards returns all sibling shard paths for path, first shard first. A
// non-sharded file returns itself.
func Shards(path string) ([]string, error) {
	m := shardRe.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}, nil
	}
	dir := filepath.Dir(path)
	var total int
	if _, err := fmt.Sscanf(m[3], "%d", &total); err != nil || total < 1 {
		return nil, fmt.Errorf("bad shard suffix in %s", path)
	}
	out := make([]string, 0, total)
	for i := 1; i <= total; i++ {
		p := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], i, total))
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("missing shard %d of %d: %s", i, total, p)
		}
		out = append(out, p)
	}
	return out, nil
}

// FromGGUF builds a Model from a GGUF file or any shard of one.
func FromGGUF(path string) (*Model, error) {
	files, err := Shards(path)
	if err != nil {
		return nil, err
	}
	headers := make([]*gguf.Header, 0, len(files))
	m := &Model{Path: files[0], Files: files}
	for _, f := range files {
		h, err := gguf.ReadHeader(f)
		if err != nil {
			return nil, err
		}
		headers = append(headers, h)
		st, err := os.Stat(f)
		if err != nil {
			return nil, err
		}
		m.WeightsBytes += uint64(st.Size())
	}
	if err := m.fill(headers, strings.TrimSuffix(filepath.Base(files[0]), ".gguf"), false); err != nil {
		return nil, fmt.Errorf("%s: %w", files[0], err)
	}
	m.Owner = ownerFromHeader(headers[0])
	if m.Owner == "" {
		m.Owner = ownerFromPath(files[0])
	}
	m.Baseline, m.BaselineNote = Classify(m.Owner, m.Name)
	return m, nil
}

// FromHeader builds a Model for a file that is not on disk, from its first
// (or only) shard's header and the total weight size reported by the host.
// partial says the header covers only the first shard, so tensor sums are
// incomplete and the parameter count falls back to a quantization estimate.
func FromHeader(h *gguf.Header, ref, name string, weightsBytes uint64, owner string, partial bool) (*Model, error) {
	m := &Model{Path: ref, Remote: true, WeightsBytes: weightsBytes}
	if err := m.fill([]*gguf.Header{h}, name, partial); err != nil {
		return nil, fmt.Errorf("%s: %w", ref, err)
	}
	if name != "" {
		m.Name = name
	}
	m.Owner = owner
	if m.Owner == "" {
		m.Owner = ownerFromHeader(h)
	}
	m.Baseline, m.BaselineNote = Classify(m.Owner, m.Name)
	return m, nil
}

// fill reads geometry from GGUF headers. headers[0] is authoritative for
// metadata; the rest only contribute tensor counts.
func (m *Model) fill(headers []*gguf.Header, fallbackName string, partial bool) error {
	h := headers[0]
	m.Arch = h.Arch()
	if m.Arch == "" {
		return errors.New("general.architecture missing")
	}
	m.Name, _ = h.String("general.name")
	if m.Name == "" {
		m.Name = fallbackName
	}
	if ft, ok := h.Uint("general.file_type"); ok {
		m.Quant = fileTypeName(ft)
	} else {
		m.Quant = "unknown"
		m.warn("general.file_type missing; quantization unknown")
	}

	for _, fh := range headers {
		for _, t := range fh.Tensors {
			n := t.Elements()
			m.TotalElems += n
			if isRoutedExpert(t.Name) {
				m.ExpertElems += n
			}
		}
	}
	switch n, ok := h.Uint("general.parameter_count"); {
	case ok && n > 0:
		m.Params, m.ParamsSource = n, Measured
	case !partial && m.TotalElems > 0:
		m.Params, m.ParamsSource = m.TotalElems, Observed
		m.warn("general.parameter_count missing; summed tensor elements instead")
	default:
		bpw, ok := BitsPerWeight(m.Quant)
		if !ok || m.WeightsBytes == 0 {
			return errors.New("cannot determine parameter count (no parameter_count, incomplete tensors, unknown quant)")
		}
		m.Params, m.ParamsSource = uint64(float64(m.WeightsBytes)*8/bpw), Inferred
		m.warn(fmt.Sprintf("parameter count estimated from %s size at %.2f bits/weight", m.Quant, bpw))
		if partial {
			// Expert accounting from one shard would be skewed; fall back.
			m.TotalElems, m.ExpertElems = 0, 0
		}
	}

	a := m.Arch
	layers, ok := h.Uint(a + ".block_count")
	if !ok {
		return fmt.Errorf("%s.block_count missing", a)
	}
	m.Layers = uint32(layers)

	kv, ok := h.UintSlice(a + ".attention.head_count_kv")
	if !ok {
		return fmt.Errorf("%s.attention.head_count_kv missing", a)
	}
	m.KVHeads = expandPerLayer(kv, m.Layers)
	if len(kv) != 1 && len(kv) != int(m.Layers) {
		m.warn(fmt.Sprintf("head_count_kv has %d entries for %d layers; padded with 0", len(kv), m.Layers))
	}
	// Hybrid architectures (qwen3next/qwen35 family) interleave recurrent
	// layers with full attention: only every Nth layer holds a KV cache. The
	// trailing next-token-prediction layers are attention layers too.
	if interval, ok := h.Uint(a + ".full_attention_interval"); ok && interval > 1 {
		nextn, _ := h.Uint(a + ".nextn_predict_layers")
		main := uint64(m.Layers) - nextn
		for i := uint64(0); i < main; i++ {
			if (i+1)%interval != 0 {
				m.KVHeads[i] = 0
			}
		}
	}

	heads, _ := h.Uint(a + ".attention.head_count")
	embd, _ := h.Uint(a + ".embedding_length")
	if k, ok := h.Uint(a + ".attention.key_length"); ok {
		m.KeyLen = uint32(k)
	} else if heads > 0 && embd > 0 {
		m.KeyLen = uint32(embd / heads)
		m.warn("attention.key_length missing; used embedding_length / head_count")
	} else {
		return fmt.Errorf("cannot determine head dimension for %s", a)
	}
	if v, ok := h.Uint(a + ".attention.value_length"); ok {
		m.ValLen = uint32(v)
	} else {
		m.ValLen = m.KeyLen
	}

	if e, ok := h.Uint(a + ".expert_count"); ok {
		m.Experts = uint32(e)
	}
	if e, ok := h.Uint(a + ".expert_used_count"); ok {
		m.ExpertsUsed = uint32(e)
	}
	return nil
}

// MLXConfig is the subset of a Hugging Face / MLX config.json fit needs.
type MLXConfig struct {
	ModelType         string   `json:"model_type"`
	Architectures     []string `json:"architectures"`
	NumHiddenLayers   uint32   `json:"num_hidden_layers"`
	NumKVHeads        uint32   `json:"num_key_value_heads"`
	NumAttentionHeads uint32   `json:"num_attention_heads"`
	HiddenSize        uint32   `json:"hidden_size"`
	HeadDim           uint32   `json:"head_dim"`
	NumExperts        uint32   `json:"num_experts"`
	NumLocalExperts   uint32   `json:"num_local_experts"`
	ExpertsPerTok     uint32   `json:"num_experts_per_tok"`
	Quantization      *struct {
		Bits int `json:"bits"`
	} `json:"quantization"`
}

// FromMLXConfig builds a Model from config.json contents plus the total size
// of the safetensors weights. Parameter count is not in config.json, so it
// is estimated from bytes and quantization bits.
func FromMLXConfig(raw []byte, ref, name string, weightsBytes uint64, owner string) (*Model, error) {
	var c MLXConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("%s: config.json: %w", ref, err)
	}
	if c.NumHiddenLayers == 0 || c.NumKVHeads == 0 {
		return nil, fmt.Errorf("%s: config.json lacks num_hidden_layers/num_key_value_heads", ref)
	}
	if weightsBytes == 0 {
		return nil, fmt.Errorf("%s: no .safetensors weights", ref)
	}
	m := &Model{Path: ref, Name: name, Arch: c.ModelType, Layers: c.NumHiddenLayers, WeightsBytes: weightsBytes, Owner: owner}
	if m.Arch == "" && len(c.Architectures) > 0 {
		m.Arch = c.Architectures[0]
	}
	m.KVHeads = expandPerLayer([]uint64{uint64(c.NumKVHeads)}, m.Layers)
	switch {
	case c.HeadDim > 0:
		m.KeyLen = c.HeadDim
	case c.NumAttentionHeads > 0 && c.HiddenSize > 0:
		m.KeyLen = c.HiddenSize / c.NumAttentionHeads
		m.warn("head_dim missing; used hidden_size / num_attention_heads")
	default:
		return nil, fmt.Errorf("%s: cannot determine head dimension", ref)
	}
	m.ValLen = m.KeyLen
	m.Experts = c.NumExperts
	if m.Experts == 0 {
		m.Experts = c.NumLocalExperts
	}
	m.ExpertsUsed = c.ExpertsPerTok
	bits := 16.0
	if c.Quantization != nil && c.Quantization.Bits > 0 {
		m.Quant = fmt.Sprintf("%d-bit", c.Quantization.Bits)
		bits = float64(c.Quantization.Bits) + 0.5 // scales and biases overhead
	} else {
		m.Quant = "bf16/f16"
	}
	m.Params = uint64(float64(m.WeightsBytes) * 8 / bits)
	m.ParamsSource = Inferred
	m.warn("parameter count estimated from file size and quantization bits")
	m.Baseline, m.BaselineNote = Classify(m.Owner, m.Name)
	return m, nil
}

// FromMLXDir builds a Model from a directory holding config.json and
// safetensors weights (the MLX / Hugging Face layout).
func FromMLXDir(dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	var bytes uint64
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") {
			if info, err := e.Info(); err == nil {
				bytes += uint64(info.Size())
				files = append(files, filepath.Join(dir, e.Name()))
			}
		}
	}
	sort.Strings(files)
	m, err := FromMLXConfig(raw, dir, filepath.Base(dir), bytes, ownerFromPath(dir))
	if err != nil {
		return nil, err
	}
	m.Files = files
	return m, nil
}

func (m *Model) warn(s string) { m.Warnings = append(m.Warnings, s) }

func expandPerLayer(v []uint64, layers uint32) []uint32 {
	out := make([]uint32, layers)
	for i := range out {
		switch {
		case len(v) == 1:
			out[i] = uint32(v[0])
		case i < len(v):
			out[i] = uint32(v[i])
		}
	}
	return out
}

func ownerFromHeader(h *gguf.Header) string {
	for _, k := range []string{
		"general.base_model.0.organization",
		"general.source.huggingface.repository",
		"general.quantized_by",
	} {
		if s, ok := h.String(k); ok && s != "" {
			if i := strings.Index(s, "/"); i > 0 {
				return s[:i]
			}
			return s
		}
	}
	return ""
}

// ownerFromPath expects the LM Studio / Hugging Face layout .../<owner>/<repo>/file.
// Content-addressed blobs (Ollama's sha256-* files) reveal nothing.
func ownerFromPath(p string) string {
	if strings.HasPrefix(filepath.Base(p), "sha256-") {
		return ""
	}
	dir := filepath.Dir(p)
	if fi, err := os.Stat(p); err == nil && fi.IsDir() {
		dir = p
	}
	owner := filepath.Base(filepath.Dir(dir))
	if owner == "." || owner == "/" || owner == "" {
		return ""
	}
	return owner
}

// baselineOwners publish base models or faithful quantizations of them.
var baselineOwners = map[string]bool{
	"qwen": true, "alibaba": true, "unsloth": true, "bartowski": true,
	"lmstudio-community": true, "ggml-org": true, "library": true, "ollama": true,
	"meta-llama": true, "google": true, "mistralai": true, "deepseek-ai": true,
	"nvidia": true, "microsoft": true, "ibm-granite": true, "zai-org": true,
	"moonshotai": true, "openai": true, "cohereforai": true, "liquidai": true,
}

// variantMarkers are words that only appear in community fine-tunes/merges.
var variantMarkers = []string{"uncensored", "heretic", "abliterated", "fusion", "merge", "roleplay", "neo-", "turbo"}

// Classify decides whether a model is a baseline or a community variant.
func Classify(owner, name string) (bool, string) {
	lower := strings.ToLower(name)
	for _, w := range variantMarkers {
		if strings.Contains(lower, w) {
			return false, fmt.Sprintf("variant (%q in name) — compare against the base model before trusting it", w)
		}
	}
	if owner == "" {
		return false, "publisher unknown — cannot confirm this is a baseline"
	}
	if baselineOwners[strings.ToLower(owner)] {
		return true, "baseline (" + owner + ")"
	}
	return false, "variant (published by " + owner + ") — compare against the base model before trusting it"
}

// bitsPerWeight is the effective storage cost of common GGUF quantizations,
// including scales and the unquantized embedding/output tensors typical of
// llama.cpp's mixed recipes. Used only when a parameter count is unavailable.
var bitsPerWeight = map[string]float64{
	"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5,
	"Q5_0": 5.5, "Q5_1": 6.0, "Q4_K_M": 4.85, "Q4_K_S": 4.6, "Q4_0": 4.55, "Q4_1": 5.0,
	"IQ4_XS": 4.3, "IQ4_NL": 4.5, "Q3_K_L": 4.3, "Q3_K_M": 3.9, "Q3_K_S": 3.5, "IQ3_M": 3.7,
	"IQ3_S": 3.5, "IQ3_XS": 3.3, "IQ3_XXS": 3.1, "Q2_K": 3.35, "Q2_K_S": 3.0, "IQ2_M": 2.7,
	"IQ2_S": 2.5, "IQ2_XS": 2.3, "IQ2_XXS": 2.1, "IQ1_M": 1.75, "IQ1_S": 1.56,
}

// BitsPerWeight returns the effective bits per parameter for a quant name.
func BitsPerWeight(quant string) (float64, bool) {
	b, ok := bitsPerWeight[strings.ToUpper(quant)]
	return b, ok
}

func fileTypeName(ft uint64) string {
	names := map[uint64]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
		10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
		16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
		22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
		28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
	}
	if n, ok := names[ft]; ok {
		return n
	}
	return fmt.Sprintf("type-%d", ft)
}
