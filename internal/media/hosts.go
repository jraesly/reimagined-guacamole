package media

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Host struct {
	Name, Path string
	Kinds      []string
}

// DetectHosts inspects only the supplied home. Unreadable locations are skipped.
// Directory symlinks are not traversed, avoiding cycles and unrelated trees.
func DetectHosts(home string) []Host {
	var out []Host
	add := func(name, path string, kinds ...string) { out = append(out, Host{name, path, kinds}) }
	// ComfyUI is looked for at most three levels below home (~/ComfyUI,
	// ~/src/ComfyUI, ~/src/ai/ComfyUI); a full home walk would take
	// seconds on a developer machine.
	const maxDepth = 3
	_ = filepath.WalkDir(home, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() {
			return nil
		}
		if d.Name() == "ComfyUI" && isDir(filepath.Join(path, "models")) {
			add("ComfyUI", path, "image", "video")
			return filepath.SkipDir
		}
		if path != home && (strings.HasPrefix(d.Name(), ".") || d.Name() == "Library" || d.Name() == "node_modules") {
			return filepath.SkipDir
		}
		if rel, err := filepath.Rel(home, path); err == nil && rel != "." && strings.Count(rel, string(filepath.Separator)) >= maxDepth-1 {
			return filepath.SkipDir
		}
		return nil
	})
	draw := filepath.Join(home, "Library", "Containers", "com.liuliu.draw-things", "Data")
	if isDir(draw) {
		add("Draw Things", draw, "image", "video")
	}
	for _, root := range WhisperRoots(home) {
		found := false
		_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if strings.HasPrefix(d.Name(), "ggml-") && filepath.Ext(path) == ".bin" && whisperMagic(path) {
				found = true
			}
			return nil
		})
		if found {
			add("whisper.cpp", root, "speech")
		}
	}
	cache := filepath.Join(home, ".cache", "huggingface", "hub")
	repos, _ := os.ReadDir(cache)
	for _, repo := range repos {
		if !repo.IsDir() || !strings.HasPrefix(repo.Name(), "models--") {
			continue
		}
		root := filepath.Join(cache, repo.Name())
		_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil || !d.IsDir() {
				return nil
			}
			if d.Name() == "blobs" {
				return filepath.SkipDir
			}
			if path == root && isDir(filepath.Join(root, "snapshots")) {
				return nil
			}
			if kind, ok := Detect(path); ok {
				if _, err := os.Stat(filepath.Join(path, "model_index.json")); err == nil {
					add("Hugging Face", path, string(kind))
				} else if kind == TTS && strings.HasPrefix(repo.Name(), "models--mlx-community--") {
					add("mlx-audio", path, string(kind))
				}
				return filepath.SkipDir
			}
			return nil
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// WhisperRoots are the supported whisper.cpp weight locations.
func WhisperRoots(home string) []string {
	return []string{filepath.Join(home, ".cache", "whisper"), filepath.Join(home, "whisper.cpp", "models"), filepath.Join(home, ".local", "share", "whisper")}
}
func isDir(path string) bool { st, err := os.Stat(path); return err == nil && st.IsDir() }
