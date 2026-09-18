package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
)

func header(arch string, extra map[string]any) []byte {
	f := gguf.Fixture{Metadata: map[string]any{
		"general.architecture":            arch,
		"general.file_type":               uint32(15),
		arch + ".block_count":             uint32(8),
		arch + ".attention.head_count":    uint32(8),
		arch + ".attention.head_count_kv": uint32(2),
		arch + ".attention.key_length":    uint32(128),
		arch + ".attention.value_length":  uint32(128),
		"tokenizer.ggml.tokens":           bigVocab(3000),
	}}
	f.Tensors = []gguf.TensorInfo{{Name: "blk.0.attn_q.weight", Dims: []uint64{1000, 1000}}}
	for k, v := range extra {
		f.Metadata[k] = v
	}
	return f.Bytes()
}

func bigVocab(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("token%06d", i)
	}
	return out
}

// serveBlob serves data with Range support and records the ranges asked for.
func serveBlob(t *testing.T, mux *http.ServeMux, path string, data []byte, ranges *[]string) {
	t.Helper()
	mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
		*ranges = append(*ranges, r.Header.Get("Range"))
		http.ServeContent(w, r, "blob", time.Time{}, bytes.NewReader(data))
	})
}

func TestParseRef(t *testing.T) {
	cases := map[string]string{
		"ollama:qwen3.8":          "ollama:qwen3.8:latest",
		"ollama:qwen3.8:27b":      "ollama:qwen3.8:27b",
		"ollama:jr/mymodel:v1":    "ollama:jr/mymodel:v1",
		"hf:unsloth/Qwen3.8-GGUF": "hf:unsloth/Qwen3.8-GGUF",
		"hf:unsloth/Q:UD-Q4_K_XL": "hf:unsloth/Q:UD-Q4_K_XL",
	}
	for in, want := range cases {
		r, ok := ParseRef(in)
		if !ok || r.String() != want {
			t.Errorf("ParseRef(%q) = %q,%v want %q", in, r, ok, want)
		}
	}
	for _, bad := range []string{"", "qwen3.8", "/path/to/file.gguf", "hf:norepo", "ollama:", "s3:bucket/key"} {
		if _, ok := ParseRef(bad); ok {
			t.Errorf("ParseRef(%q) should fail", bad)
		}
	}
}

func TestFetchHeaderGrowsRange(t *testing.T) {
	data := append(header("qwen35", nil), make([]byte, 5000)...)
	var ranges []string
	mux := http.NewServeMux()
	serveBlob(t, mux, "/blob", data, &ranges)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	old := InitialRange
	InitialRange = 4096
	defer func() { InitialRange = old }()

	h, err := FetchHeader(context.Background(), srv.URL+"/blob")
	if err != nil {
		t.Fatal(err)
	}
	if h.Arch() != "qwen35" {
		t.Errorf("arch = %q", h.Arch())
	}
	if len(ranges) < 2 || ranges[0] != "bytes=0-4095" || ranges[1] != "bytes=0-8191" {
		t.Errorf("ranges = %v; expected the range to double after a truncated parse", ranges)
	}
}

func TestFetchHeaderServerIgnoresRange(t *testing.T) {
	data := append(header("qwen35", nil), make([]byte, 100)...)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(data)
	}))
	defer srv.Close()
	h, err := FetchHeader(context.Background(), srv.URL)
	if err != nil || h.Arch() != "qwen35" {
		t.Fatalf("h=%v err=%v", h, err)
	}
}

func TestFetchHeaderErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/404":
			w.WriteHeader(404)
		case "/notgguf":
			http.ServeContent(w, r, "x", time.Time{}, strings.NewReader("this is not a model file at all"))
		}
	}))
	defer srv.Close()
	if _, err := FetchHeader(context.Background(), srv.URL+"/404"); err == nil || !strings.Contains(err.Error(), "HTTP 404") {
		t.Errorf("404: %v", err)
	}
	if _, err := FetchHeader(context.Background(), srv.URL+"/notgguf"); err == nil || !strings.Contains(err.Error(), "not a GGUF") {
		t.Errorf("notgguf: %v", err)
	}
}

func TestResolveOllama(t *testing.T) {
	blob := append(header("qwen35", map[string]any{"general.parameter_count": uint64(27_300_000_000)}), make([]byte, 10)...)
	var ranges []string
	mux := http.NewServeMux()
	mux.HandleFunc("/v2/library/qwen3.8/manifests/27b", func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Accept"), "manifest.v2") {
			t.Errorf("missing Accept header: %q", r.Header.Get("Accept"))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"layers": []map[string]any{
			{"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:pp", "size": 931146016},
			{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:mm", "size": 16810714464},
		}})
	})
	mux.HandleFunc("/v2/library/cloudonly/manifests/latest", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"layers": []map[string]any{}})
	})
	serveBlob(t, mux, "/v2/library/qwen3.8/blobs/sha256:mm", blob, &ranges)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	OllamaRegistry = srv.URL

	m, err := Resolve(context.Background(), Ref{Host: "ollama", Owner: "library", Name: "qwen3.8", Tag: "27b"})
	if err != nil {
		t.Fatal(err)
	}
	if !m.Remote || m.Name != "qwen3.8:27b" || m.WeightsBytes != 16810714464 || m.Params != 27_300_000_000 || !m.Baseline {
		t.Errorf("model = %+v", m)
	}
	if len(m.Warnings) == 0 || !strings.Contains(m.Warnings[len(m.Warnings)-1], "vision projector") {
		t.Errorf("expected projector warning, got %v", m.Warnings)
	}
	if len(ranges) == 0 || !strings.HasPrefix(ranges[0], "bytes=0-") {
		t.Errorf("blob was not range-requested: %v", ranges)
	}

	if _, err := Resolve(context.Background(), Ref{Host: "ollama", Owner: "library", Name: "cloudonly", Tag: "latest"}); err == nil || !strings.Contains(err.Error(), "no model layer") {
		t.Errorf("cloud-only: %v", err)
	}
	if _, err := Resolve(context.Background(), Ref{Host: "ollama", Owner: "library", Name: "missing", Tag: "latest"}); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("missing: %v", err)
	}
}

func TestResolveHFGGUFShardedAndFilter(t *testing.T) {
	blob := append(header("qwen35", nil), make([]byte, 10)...)
	var ranges []string
	mux := http.NewServeMux()
	tree := []map[string]any{
		{"type": "file", "path": "README.md", "size": 10},
		{"type": "file", "path": "mmproj-F16.gguf", "size": 999},
		{"type": "file", "path": "Q-UD-Q4_K_XL-00001-of-00002.gguf", "size": 1000},
		{"type": "file", "path": "Q-UD-Q4_K_XL-00002-of-00002.gguf", "size": 2000},
		{"type": "file", "path": "Q-Q8_0.gguf", "size": 5000},
	}
	mux.HandleFunc("/api/models/unsloth/Q/tree/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(tree)
	})
	serveBlob(t, mux, "/unsloth/Q/resolve/main/Q-UD-Q4_K_XL-00001-of-00002.gguf", blob, &ranges)
	serveBlob(t, mux, "/unsloth/Q/resolve/main/Q-Q8_0.gguf", blob, &ranges)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	HFBase = srv.URL

	// Default preference picks Q4_K_XL (sharded): total bytes = both shards,
	// header from shard 1, params estimated because the header is partial.
	m, err := Resolve(context.Background(), Ref{Host: "hf", Owner: "unsloth", Name: "Q"})
	if err != nil {
		t.Fatal(err)
	}
	if m.WeightsBytes != 3000 || !strings.HasSuffix(m.Name, "00001-of-00002.gguf") || m.ParamsSource != "inferred" {
		t.Errorf("model = %+v", m)
	}
	// Explicit filter picks the Q8_0 single file.
	m, err = Resolve(context.Background(), Ref{Host: "hf", Owner: "unsloth", Name: "Q", Tag: "q8_0"})
	if err != nil {
		t.Fatal(err)
	}
	if m.WeightsBytes != 5000 || m.ParamsSource != "observed" {
		t.Errorf("q8 model = %+v", m)
	}
	if _, err := Resolve(context.Background(), Ref{Host: "hf", Owner: "unsloth", Name: "Q", Tag: "IQ1_S"}); err == nil || !strings.Contains(err.Error(), "available:") {
		t.Errorf("bad filter: %v", err)
	}
}

func TestResolveHFMLX(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/mlx-community/M/tree/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"type": "file", "path": "config.json", "size": 500},
			{"type": "file", "path": "model-00001-of-00002.safetensors", "size": 4000},
			{"type": "file", "path": "model-00002-of-00002.safetensors", "size": 500},
		})
	})
	mux.HandleFunc("/mlx-community/M/resolve/main/config.json", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model_type":"qwen3","num_hidden_layers":4,"num_key_value_heads":2,"head_dim":64,"quantization":{"bits":4}}`))
	})
	mux.HandleFunc("/api/models/x/empty/tree/main", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{{"type": "file", "path": "README.md", "size": 1}})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	HFBase = srv.URL

	m, err := Resolve(context.Background(), Ref{Host: "hf", Owner: "mlx-community", Name: "M"})
	if err != nil {
		t.Fatal(err)
	}
	if !m.Remote || m.WeightsBytes != 4500 || m.Layers != 4 || m.Quant != "4-bit" {
		t.Errorf("model = %+v", m)
	}
	if _, err := Resolve(context.Background(), Ref{Host: "hf", Owner: "x", Name: "empty"}); err == nil || !strings.Contains(err.Error(), "no .gguf") {
		t.Errorf("empty repo: %v", err)
	}
}
