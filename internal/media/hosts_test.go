package media

import (
	"path/filepath"
	"testing"
)

func TestDetectHosts(t *testing.T) {
	home := t.TempDir()
	put(t, filepath.Join(home, "work", "ComfyUI", "models", "checkpoints", "placeholder"), nil)
	put(t, filepath.Join(home, "Library", "Containers", "com.liuliu.draw-things", "Data", "Documents", "Models", "placeholder"), nil)
	put(t, filepath.Join(home, ".cache", "whisper", "ggml-tiny.bin"), []byte("lmgg"))
	put(t, filepath.Join(home, ".cache", "huggingface", "hub", "models--mlx-community--kokoro", "snapshots", "rev", "config.json"), []byte(`{"model_type":"kokoro"}`))
	put(t, filepath.Join(home, ".cache", "huggingface", "hub", "models--owner--sdxl", "snapshots", "rev", "model_index.json"), []byte(`{"_class_name":"StableDiffusionXLPipeline"}`))
	hosts := DetectHosts(home)
	names := map[string]bool{}
	for _, h := range hosts {
		names[h.Name] = true
		if len(h.Kinds) == 0 {
			t.Fatalf("missing kinds: %+v", h)
		}
	}
	for _, name := range []string{"ComfyUI", "Draw Things", "mlx-audio", "whisper.cpp", "Hugging Face"} {
		if !names[name] {
			t.Errorf("missing %s: %+v", name, hosts)
		}
	}
	if got := DetectHosts(t.TempDir()); len(got) != 0 {
		t.Fatalf("empty home: %+v", got)
	}
}
