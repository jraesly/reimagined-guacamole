package scan

import (
	"os"
	"path/filepath"
	"testing"
)

func touch(t *testing.T, p string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestHFLayoutFindsGGUFMLXAndExtras(t *testing.T) {
	root := t.TempDir()
	touch(t, filepath.Join(root, "DavidAU", "Qwen-GGUF", "q4.gguf"), "x")
	touch(t, filepath.Join(root, "DavidAU", "Qwen-GGUF", "mmproj-F32.gguf"), "x")
	touch(t, filepath.Join(root, "mlx-community", "Qwen-4bit", "config.json"), "{}")
	touch(t, filepath.Join(root, "mlx-community", "Qwen-4bit", "model.safetensors"), "x")
	touch(t, filepath.Join(root, "unsloth", "Big-GGUF", "big-00001-of-00003.gguf"), "x")
	touch(t, filepath.Join(root, "unsloth", "Big-GGUF", "big-00002-of-00003.gguf"), "x")
	touch(t, filepath.Join(root, "unsloth", "Big-GGUF", "big-00003-of-00003.gguf"), "x")
	found := HFLayout(root, "lmstudio")
	if len(found) != 3 {
		t.Fatalf("found %d entries: %+v", len(found), found)
	}
	byOwner := map[string]Found{}
	for _, f := range found {
		byOwner[f.Owner] = f
	}
	if d := byOwner["DavidAU"]; len(d.Extras) != 1 || d.IsDir || d.Names[0] != "DavidAU/Qwen-GGUF" {
		t.Errorf("DavidAU = %+v", d)
	}
	if m := byOwner["mlx-community"]; !m.IsDir {
		t.Errorf("mlx = %+v", m)
	}
	if u := byOwner["unsloth"]; filepath.Base(u.Path) != "big-00001-of-00003.gguf" {
		t.Errorf("sharded = %+v", u)
	}
}

func TestHFLayoutMissingRoot(t *testing.T) {
	if got := HFLayout(filepath.Join(t.TempDir(), "nope"), "lmstudio"); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestOllamaResolvesTagsToBlobs(t *testing.T) {
	root := t.TempDir()
	blob := filepath.Join(root, "blobs", "sha256-abc")
	touch(t, blob, "weights")
	manifest := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc"},{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:zzz"}]}`
	touch(t, filepath.Join(root, "manifests", "registry.ollama.ai", "library", "qwen3.8", "27b"), manifest)
	touch(t, filepath.Join(root, "manifests", "registry.ollama.ai", "library", "qwen3.8", "27b-32k"), manifest)
	touch(t, filepath.Join(root, "manifests", "registry.ollama.ai", "library", "gone", "latest"),
		`{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:missing"}]}`)
	found, err := Ollama(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(found) != 2 {
		t.Fatalf("found = %+v", found)
	}
	for _, f := range found {
		if f.Path != blob || f.Owner != "library" || f.Source != "ollama" {
			t.Errorf("entry = %+v", f)
		}
	}
}

func TestOllamaMissingRoot(t *testing.T) {
	found, err := Ollama(filepath.Join(t.TempDir(), "nope"))
	if err != nil || len(found) != 0 {
		t.Errorf("found=%v err=%v", found, err)
	}
}

func TestAllMergesDuplicateBlobs(t *testing.T) {
	home := t.TempDir()
	roots := Roots(home)
	blob := filepath.Join(roots["ollama"], "blobs", "sha256-abc")
	touch(t, blob, "w")
	manifest := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc"}]}`
	touch(t, filepath.Join(roots["ollama"], "manifests", "r", "library", "m", "a"), manifest)
	touch(t, filepath.Join(roots["ollama"], "manifests", "r", "library", "m", "b"), manifest)
	touch(t, filepath.Join(roots["lmstudio"], "o", "r", "q.gguf"), "w")
	touch(t, filepath.Join(roots["hfcache"], "models--o2--r2", "snapshots", "deadbeef", "q.gguf"), "w")
	all, err := All(home)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 {
		t.Fatalf("all = %+v", all)
	}
	for _, f := range all {
		if f.Path == blob && (len(f.Names) != 2 || f.Names[0] != "m:a" || f.Names[1] != "m:b") {
			t.Errorf("merged names = %v", f.Names)
		}
		if f.Source == "hfcache" && f.Owner != "o2" {
			t.Errorf("hfcache owner = %q", f.Owner)
		}
	}
}
