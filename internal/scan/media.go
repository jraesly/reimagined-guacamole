package scan

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jraesly/reimagined-guacamole/internal/media"
)

type MediaFound struct{ Path, Name, Kind, Source string }

// Media discovers candidates separately from the LLM scanner. A detected model
// directory is emitted once; its individual components are not separate models.
// Missing/unreadable paths are skipped, as this best-effort API has no error result.
func Media(home string) []MediaFound {
	byPath := map[string]MediaFound{}
	walk := func(root, source string) {
		_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() && d.Name() == "blobs" {
				return filepath.SkipDir
			}
			// A cache repository is a container; named XTTS repositories must
			// resolve to their snapshot rather than trigger the name heuristic.
			if d.IsDir() && strings.HasPrefix(d.Name(), "models--") {
				if st, err := os.Stat(filepath.Join(path, "snapshots")); err == nil && st.IsDir() {
					return nil
				}
			}
			if kind, ok := media.Detect(path); ok {
				byPath[path] = MediaFound{Path: path, Name: filepath.Base(path), Kind: string(kind), Source: source}
				if d.IsDir() {
					return filepath.SkipDir
				}
			}
			return nil
		})
	}
	for _, host := range media.DetectHosts(home) {
		switch host.Name {
		case "ComfyUI":
			for _, sub := range []string{"checkpoints", "unet", "diffusion_models", "vae", "clip"} {
				walk(filepath.Join(host.Path, "models", sub), "comfyui")
			}
		case "Draw Things":
			walk(host.Path, "draw-things")
		}
	}
	walk(filepath.Join(home, ".cache", "huggingface", "hub"), "hfcache")
	for _, root := range media.WhisperRoots(home) {
		walk(root, "whisper.cpp")
	}
	out := make([]MediaFound, 0, len(byPath))
	for _, m := range byPath {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}
