package media

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
)

func put(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func safeFixture(t *testing.T, path string, tensors ...Tensor) {
	t.Helper()
	obj := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var offset uint64
	for _, tensor := range tensors {
		obj[tensor.Name] = map[string]any{"dtype": tensor.DType, "shape": tensor.Shape, "data_offsets": []uint64{offset, offset + tensor.Bytes}}
		offset += tensor.Bytes
	}
	raw, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(raw)))
	data = append(data, raw...)
	data = append(data, make([]byte, offset)...)
	put(t, path, data)
}

func TestReadHeader(t *testing.T) {
	p := filepath.Join(t.TempDir(), "model.safetensors")
	safeFixture(t, p, Tensor{"a", "BF16", []uint64{2}, 4}, Tensor{"b", "F32", []uint64{2}, 8})
	h, err := ReadHeader(p)
	if err != nil {
		t.Fatal(err)
	}
	if h.TotalBytes() != 12 || h.ByDType()["BF16"] != 4 || h.Metadata["format"] != "pt" {
		t.Fatalf("bad header: %+v", h)
	}
}

func TestReadHeaderRejectsMalformed(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		n         uint64
	}{
		{"truncated", "{}", 20}, {"oversize", "", 100 << 20},
		{"negative", `{"x":{"dtype":"F16","shape":[1],"data_offsets":[-1,2]}}`, 0},
		{"reversed", `{"x":{"dtype":"F16","shape":[1],"data_offsets":[4,2]}}`, 0},
		{"missing_offsets", `{"x":{"dtype":"F16","shape":[1]}}`, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.n
			if n == 0 {
				n = uint64(len(tc.raw))
			}
			data := make([]byte, 8)
			binary.LittleEndian.PutUint64(data, n)
			p := filepath.Join(t.TempDir(), "bad.safetensors")
			put(t, p, append(data, tc.raw...))
			if _, err := ReadHeader(p); err == nil {
				t.Fatal("accepted invalid header")
			}
		})
	}
}

func TestReadShardedDir(t *testing.T) {
	dir := t.TempDir()
	safeFixture(t, filepath.Join(dir, "a.safetensors"), Tensor{"a", "F16", []uint64{2}, 4}, Tensor{"b", "F16", []uint64{2}, 4})
	safeFixture(t, filepath.Join(dir, "b.safetensors"), Tensor{"c", "BF16", []uint64{3}, 6})
	put(t, filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{"a":"a.safetensors","b":"a.safetensors","c":"b.safetensors"}}`))
	h, err := ReadShardedDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if h.TotalBytes() != 14 || len(h.Tensors) != 3 {
		t.Fatalf("bad shard sum: %+v", h)
	}
}

func TestLoadDiffusers(t *testing.T) {
	for class, family := range map[string]string{"StableDiffusionPipeline": "sd15", "StableDiffusionXLPipeline": "sdxl", "StableDiffusion3Pipeline": "sd3", "FluxPipeline": "flux", "WanPipeline": "wan", "HunyuanVideoPipeline": "hunyuan-video", "LTXPipeline": "ltx", "NewPipeline": "unknown"} {
		t.Run(family, func(t *testing.T) {
			dir := t.TempDir()
			raw, _ := json.Marshal(map[string]any{"_class_name": class, "unet": []string{"diffusers", "UNet"}, "text_encoder": []string{"transformers", "Text"}, "vae": []string{"diffusers", "VAE"}})
			put(t, filepath.Join(dir, "model_index.json"), raw)
			for _, name := range []string{"unet", "text_encoder", "vae"} {
				safeFixture(t, filepath.Join(dir, name, "model.safetensors"), Tensor{"w", "BF16", []uint64{2}, 4})
			}
			m, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			wantKind := Image
			if family == "wan" || family == "hunyuan-video" || family == "ltx" {
				wantKind = Video
			}
			if m.Family != family || m.Kind != wantKind || m.WeightsBytes != 12 || len(m.Components) != 3 {
				t.Fatalf("bad model: %+v", m)
			}
			if family == "unknown" && len(m.Warnings) == 0 {
				t.Fatal("missing warning")
			}
		})
	}
}

func TestLoadSingleFile(t *testing.T) {
	for _, tc := range []struct {
		family string
		names  []string
	}{
		{"flux", []string{"double_blocks.0.weight"}},
		{"sdxl", []string{"model.diffusion_model.input_blocks.0", "conditioner.embedders.0"}},
		{"sd15", []string{"model.diffusion_model.input_blocks.0", "cond_stage_model.weight"}},
		{"sd3", []string{"model.diffusion_model.joint_blocks.0"}},
		{"unknown", []string{"weight"}},
	} {
		t.Run(tc.family, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "model.safetensors")
			var tensors []Tensor
			for _, n := range tc.names {
				tensors = append(tensors, Tensor{n, "F16", []uint64{2}, 4})
			}
			safeFixture(t, p, tensors...)
			m, err := Load(p)
			if err != nil {
				t.Fatal(err)
			}
			if m.Family != tc.family || m.Kind != Image {
				t.Fatalf("%+v", m)
			}
		})
	}
}

func TestLoadGGUF(t *testing.T) {
	p := filepath.Join(t.TempDir(), "flux.gguf")
	raw := (gguf.Fixture{Metadata: map[string]any{"general.architecture": "flux", "general.file_type": uint32(15)}}).Bytes()
	put(t, p, raw)
	m, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.Family != "flux" || m.WeightsBytes != uint64(len(raw)) || m.Components[0].DType != "Q4_K_M" {
		t.Fatalf("%+v", m)
	}
}

func TestAudioDetection(t *testing.T) {
	for _, tc := range []struct {
		config, family string
		kind           Kind
	}{
		{`{"model_type":"kokoro"}`, "kokoro", TTS}, {`{"model":"xtts"}`, "xtts", TTS},
		{`{"model_type":"csm"}`, "csm", TTS}, {`{"model_type":"bark"}`, "bark", TTS}, {`{"model_type":"sesame"}`, "sesame", TTS},
		{`{"model_type":"whisper"}`, "whisper", Speech},
	} {
		t.Run(tc.family, func(t *testing.T) {
			dir := t.TempDir()
			put(t, filepath.Join(dir, "config.json"), []byte(tc.config))
			safeFixture(t, filepath.Join(dir, "model.safetensors"), Tensor{"w", "F16", []uint64{2}, 4})
			kind, ok := Detect(dir)
			if !ok || kind != tc.kind {
				t.Fatalf("detect %s %v", kind, ok)
			}
			m, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			if m.Family != tc.family || m.WeightsBytes != 4 {
				t.Fatalf("%+v", m)
			}
		})
	}
	for _, valid := range []bool{true, false} {
		p := filepath.Join(t.TempDir(), "ggml-tiny.bin")
		raw := []byte("bad!")
		if valid {
			raw = []byte("lmgg")
		}
		put(t, p, raw)
		_, err := Load(p)
		_, ok := Detect(p)
		if (err == nil) != valid || ok != valid {
			t.Fatalf("valid=%v err=%v detect=%v", valid, err, ok)
		}
	}
	for _, ext := range []string{".onnx", ".pth", ".safetensors"} {
		p := filepath.Join(t.TempDir(), "kokoro-v1"+ext)
		if ext == ".safetensors" {
			safeFixture(t, p, Tensor{"w", "F16", []uint64{2}, 4})
		} else {
			put(t, p, []byte("weights"))
		}
		m, err := Load(p)
		if err != nil || m.Family != "kokoro" {
			t.Fatalf("%+v %v", m, err)
		}
	}
}

func TestFit(t *testing.T) {
	for _, tc := range []struct {
		name               string
		weights            uint64
		activation, budget float64
		verdict            string
	}{
		{"yes", 4, 1, 10, "yes"}, {"boundary_yes", 8, 1, 10, "yes"}, {"tight", 8, 2, 10, "tight"}, {"no", 9, 2, 10, "no"},
		{"weights_only", 10, 0, 10, "weights-only"}, {"weights_exceed", 11, 0, 10, "no"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := Fit(&Model{WeightsBytes: tc.weights << 30}, tc.budget, Options{ActivationGB: tc.activation, Note: "local run"})
			if r.Verdict != tc.verdict || r.TotalGB != float64(tc.weights)+tc.activation || r.Basis == "" {
				t.Fatalf("%+v", r)
			}
		})
	}
}

func TestHeaderOnly(t *testing.T) {
	p := filepath.Join(t.TempDir(), "header.safetensors")
	safeFixture(t, p, Tensor{"w", "BF16", []uint64{2}, 4})
	st, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Truncate(p, st.Size()-4); err != nil {
		t.Fatal(err)
	}
	h, err := ReadHeader(p)
	if err != nil || h.TotalBytes() != 4 {
		t.Fatalf("header-only read: %+v %v", h, err)
	}
}

func TestShardedErrors(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"missing", `{"weight_map":{"w":"missing.safetensors"}}`},
		{"traversal", `{"weight_map":{"w":"../outside.safetensors"}}`},
		{"empty", `{"weight_map":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			put(t, filepath.Join(dir, "model.safetensors.index.json"), []byte(tc.raw))
			if _, err := ReadShardedDir(dir); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestDiffusersShardedComponent(t *testing.T) {
	dir := t.TempDir()
	put(t, filepath.Join(dir, "model_index.json"), []byte(`{"_class_name":"FluxPipeline","transformer":["diffusers","Transformer"],"optional":[null,null]}`))
	sub := filepath.Join(dir, "transformer")
	put(t, filepath.Join(sub, "config.json"), []byte(`{"_class_name":"FluxTransformer2DModel"}`))
	put(t, filepath.Join(sub, "diffusion_pytorch_model.safetensors.index.json"), []byte(`{"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}`))
	safeFixture(t, filepath.Join(sub, "a.safetensors"), Tensor{"a", "BF16", []uint64{2}, 4})
	safeFixture(t, filepath.Join(sub, "b.safetensors"), Tensor{"b", "F8_E4M3", []uint64{2}, 2})
	safeFixture(t, filepath.Join(sub, "unused.safetensors"), Tensor{"unused", "F32", []uint64{2}, 8})
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if m.WeightsBytes != 6 || len(m.Components) != 1 || m.Components[0].ClassName != "FluxTransformer2DModel" {
		t.Fatalf("%+v", m)
	}
}

func TestDetectRejectsLLMGGUF(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llama.gguf")
	put(t, p, (gguf.Fixture{Metadata: map[string]any{"general.architecture": "llama"}}).Bytes())
	if _, ok := Detect(p); ok {
		t.Fatal("LLM detected as media")
	}
	if _, err := Load(p); err == nil {
		t.Fatal("LLM loaded as media")
	}
}

func TestFitInvalidInputs(t *testing.T) {
	for _, value := range []float64{-1, math.NaN(), math.Inf(1)} {
		if r := Fit(&Model{}, 10, Options{ActivationGB: value}); r.Verdict != "no" {
			t.Fatalf("%+v", r)
		}
		if r := Fit(&Model{}, value, Options{}); r.Verdict != "no" {
			t.Fatalf("%+v", r)
		}
	}
	if r := Fit(nil, 10, Options{}); r.Verdict != "no" {
		t.Fatalf("%+v", r)
	}
}

func TestFamilyNote(t *testing.T) {
	for _, family := range []string{"sd15", "sd1", "sdxl", "sd3", "flux", "wan", "hunyuan-video", "ltx", "kokoro", "xtts", "csm", "bark", "sesame", "whisper", "unknown"} {
		note := FamilyNote(family)
		if note == "" {
			t.Errorf("missing note for %s", family)
		}
	}
}
