package scan

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestMedia(t *testing.T) {
	home := t.TempDir()
	write := func(rel string, raw []byte) {
		t.Helper()
		path := filepath.Join(home, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0644); err != nil {
			t.Fatal(err)
		}
	}
	raw := []byte(`{"double_blocks.0":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`)
	safe := make([]byte, 8)
	binary.LittleEndian.PutUint64(safe, uint64(len(raw)))
	safe = append(safe, raw...)
	safe = append(safe, make([]byte, 4)...)
	write("ComfyUI/models/checkpoints/flux.safetensors", safe)
	write("Library/Containers/com.liuliu.draw-things/Data/Documents/Models/sd.safetensors", safe)
	write(".cache/whisper/ggml-tiny.bin", []byte("lmgg"))
	write(".cache/whisper/ggml-invalid.bin", []byte("bad!"))
	root := ".cache/huggingface/hub/models--owner--sdxl/snapshots/rev/"
	write(root+"model_index.json", []byte(`{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet"]}`))
	write(root+"unet/model.safetensors", safe)
	write(".cache/huggingface/hub/models--mlx-community--kokoro/snapshots/rev/config.json", []byte(`{"model_type":"kokoro"}`))
	write(".cache/huggingface/hub/models--mlx-community--kokoro/snapshots/rev/model.safetensors", safe)
	write(".cache/huggingface/hub/models--owner--single/snapshots/rev/flux.safetensors", safe)
	got := Media(home)
	if len(got) != 6 {
		t.Fatalf("want 6 found %d: %+v", len(got), got)
	}
	kinds := map[string]int{}
	for _, m := range got {
		kinds[m.Kind]++
		if m.Source == "" || m.Name == "" {
			t.Fatalf("missing identity: %+v", m)
		}
	}
	if kinds["image"] != 4 || kinds["tts"] != 1 || kinds["speech"] != 1 {
		t.Fatalf("kinds: %v", kinds)
	}
	if got := Media(t.TempDir()); len(got) != 0 {
		t.Fatalf("empty home: %+v", got)
	}
}

func TestMediaHFLinks(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(home, ".cache", "huggingface", "hub", "models--mlx-community--whisper")
	snap := filepath.Join(repo, "snapshots", "rev")
	if err := os.MkdirAll(snap, 0755); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(repo, "blobs")
	if err := os.MkdirAll(blob, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blob, "config"), []byte(`{"model_type":"whisper"}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(blob, "config"), filepath.Join(snap, "config.json")); err != nil {
		t.Fatal(err)
	}
	got := Media(home)
	if len(got) != 1 || got[0].Path != snap || got[0].Kind != "speech" {
		t.Fatalf("%+v", got)
	}
}

func TestMediaXTTSCacheSnapshot(t *testing.T) {
	home := t.TempDir()
	snap := filepath.Join(home, ".cache", "huggingface", "hub", "models--owner--xtts", "snapshots", "rev")
	if err := os.MkdirAll(snap, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snap, "config.json"), []byte(`{"model":"xtts"}`), 0644); err != nil {
		t.Fatal(err)
	}
	got := Media(home)
	if len(got) != 1 || got[0].Path != snap {
		t.Fatalf("want snapshot, got %+v", got)
	}
}
