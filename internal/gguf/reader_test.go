package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func denseFixture() Fixture {
	return Fixture{
		Metadata: map[string]any{
			"general.architecture":              "qwen35",
			"general.name":                      "Qwen3.8 27B",
			"general.parameter_count":           uint64(27_300_000_000),
			"general.file_type":                 uint32(15),
			"qwen35.block_count":                uint32(66),
			"qwen35.attention.head_count":       uint32(40),
			"qwen35.attention.head_count_kv":    uint32(8),
			"qwen35.embedding_length":           uint32(5120),
			"qwen35.attention.key_length":       uint32(128),
			"qwen35.attention.value_length":     uint32(128),
			"tokenizer.ggml.tokens":             []string{"a", "b"},
			"general.base_model.0.organization": "Qwen",
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{5120, 152064}, Type: 15},
			{Name: "blk.0.attn_q.weight", Dims: []uint64{5120, 5120}, Type: 15},
		},
	}
}

func TestParseRoundTrip(t *testing.T) {
	h, err := Parse(bytes.NewReader(denseFixture().Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if h.Version != 3 {
		t.Errorf("version = %d, want 3", h.Version)
	}
	if h.Arch() != "qwen35" {
		t.Errorf("arch = %q", h.Arch())
	}
	if n, _ := h.Uint("general.parameter_count"); n != 27_300_000_000 {
		t.Errorf("parameter_count = %d", n)
	}
	if kv, _ := h.UintSlice("qwen35.attention.head_count_kv"); len(kv) != 1 || kv[0] != 8 {
		t.Errorf("head_count_kv = %v, want [8]", kv)
	}
	if len(h.Tensors) != 2 {
		t.Fatalf("tensors = %d, want 2", len(h.Tensors))
	}
	if got := h.Tensors[0].Elements(); got != 5120*152064 {
		t.Errorf("elements = %d", got)
	}
	if toks, ok := h.Metadata["tokenizer.ggml.tokens"].([]any); !ok || len(toks) != 2 {
		t.Errorf("string array not parsed: %v", h.Metadata["tokenizer.ggml.tokens"])
	}
}

func TestPerLayerKVHeadsArray(t *testing.T) {
	f := denseFixture()
	// Hybrid architectures store one head_count_kv per layer, with 0 for
	// recurrent (non-attention) layers.
	f.Metadata["qwen35.attention.head_count_kv"] = []uint32{0, 0, 0, 8, 0, 0, 0, 8}
	f.Metadata["qwen35.block_count"] = uint32(8)
	h, err := Parse(bytes.NewReader(f.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	kv, ok := h.UintSlice("qwen35.attention.head_count_kv")
	if !ok || len(kv) != 8 {
		t.Fatalf("head_count_kv = %v", kv)
	}
	sum := uint64(0)
	for _, v := range kv {
		sum += v
	}
	if sum != 16 {
		t.Errorf("sum of kv heads = %d, want 16", sum)
	}
}

func TestReadHeaderFromFileIgnoresTrailingData(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	data := append(denseFixture().Bytes(), bytes.Repeat([]byte{0xAB}, 4096)...)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := ReadHeader(path)
	if err != nil {
		t.Fatal(err)
	}
	if h.TensorCount != 2 {
		t.Errorf("tensor count = %d", h.TensorCount)
	}
}

func TestParseRejectsBadMagic(t *testing.T) {
	_, err := Parse(bytes.NewReader([]byte("NOPE\x03\x00\x00\x00")))
	if err == nil || !strings.Contains(err.Error(), "not a GGUF") {
		t.Errorf("err = %v", err)
	}
}

func TestParseRejectsUnsupportedVersion(t *testing.T) {
	f := denseFixture()
	f.Version = 1
	_, err := Parse(bytes.NewReader(f.Bytes()))
	if err == nil || !strings.Contains(err.Error(), "version") {
		t.Errorf("err = %v", err)
	}
}

func TestParseTruncated(t *testing.T) {
	full := denseFixture().Bytes()
	for _, cut := range []int{4, 12, 24, len(full) / 2, len(full) - 1} {
		_, err := Parse(bytes.NewReader(full[:cut]))
		if err == nil {
			t.Errorf("cut at %d: expected error", cut)
		}
	}
}

func TestParseRejectsImplausibleCounts(t *testing.T) {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint32(magic))
	_ = binary.Write(&b, binary.LittleEndian, uint32(3))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1<<40)) // tensor count
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	_, err := Parse(&b)
	if err == nil || !strings.Contains(err.Error(), "implausible") {
		t.Errorf("err = %v", err)
	}
}

func TestParseRejectsHugeString(t *testing.T) {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint32(magic))
	_ = binary.Write(&b, binary.LittleEndian, uint32(3))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1<<30)) // key length
	_, err := Parse(&b)
	if err == nil || !strings.Contains(err.Error(), "exceeds limit") {
		t.Errorf("err = %v", err)
	}
}

func TestUintAcceptsAllIntegerWidths(t *testing.T) {
	f := Fixture{Metadata: map[string]any{
		"a": uint8(1), "b": uint16(2), "c": uint32(3), "d": uint64(4), "e": int32(5), "f": int64(6),
		"neg": int32(-1),
	}}
	h, err := Parse(bytes.NewReader(f.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]uint64{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6} {
		if got, ok := h.Uint(k); !ok || got != want {
			t.Errorf("%s = %d,%v want %d", k, got, ok, want)
		}
	}
	if _, ok := h.Uint("neg"); ok {
		t.Error("negative value should not convert to uint")
	}
	if _, ok := h.Uint("missing"); ok {
		t.Error("missing key should report !ok")
	}
}
