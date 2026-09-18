// Package scan finds model files in the places local servers keep them.
package scan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Found is a model file plus where it came from.
type Found struct {
	Path   string   // GGUF file (first shard) or MLX directory
	Names  []string // every name that resolves to this path (Ollama tags, repo names)
	Owner  string   // publisher/namespace when the layout reveals it
	Source string   // "lmstudio", "ollama", "hfcache"
	IsDir  bool     // MLX / safetensors directory
	Extras []string // companion files that are not weights, e.g. mmproj
}

// Roots lists the directories scanned, resolved against home.
func Roots(home string) map[string]string {
	return map[string]string{
		"lmstudio":  filepath.Join(home, ".lmstudio", "models"),
		"lmstudio2": filepath.Join(home, ".cache", "lm-studio", "models"),
		"ollama":    filepath.Join(home, ".ollama", "models"),
		"hfcache":   filepath.Join(home, ".cache", "huggingface", "hub"),
	}
}

// All scans every known root under home and merges duplicates by path.
func All(home string) ([]Found, error) {
	roots := Roots(home)
	byPath := map[string]*Found{}
	add := func(f Found) {
		if cur, ok := byPath[f.Path]; ok {
			cur.Names = append(cur.Names, f.Names...)
			if cur.Owner == "" {
				cur.Owner = f.Owner
			}
			return
		}
		c := f
		byPath[f.Path] = &c
	}
	for _, key := range []string{"lmstudio", "lmstudio2"} {
		for _, f := range HFLayout(roots[key], "lmstudio") {
			add(f)
		}
	}
	for _, f := range HFCache(roots["hfcache"]) {
		add(f)
	}
	ol, err := Ollama(roots["ollama"])
	if err != nil {
		return nil, err
	}
	for _, f := range ol {
		add(f)
	}
	out := make([]Found, 0, len(byPath))
	for _, f := range byPath {
		sort.Strings(f.Names)
		f.Names = dedupe(f.Names)
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// HFLayout scans <root>/<owner>/<repo>/ for GGUF files and MLX directories.
func HFLayout(root, source string) []Found {
	var out []Found
	owners, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	for _, o := range owners {
		if !o.IsDir() {
			continue
		}
		repos, err := os.ReadDir(filepath.Join(root, o.Name()))
		if err != nil {
			continue
		}
		for _, r := range repos {
			if !r.IsDir() {
				continue
			}
			dir := filepath.Join(root, o.Name(), r.Name())
			out = append(out, inRepoDir(dir, o.Name(), r.Name(), source)...)
		}
	}
	return out
}

// inRepoDir classifies the files of one model directory.
func inRepoDir(dir, owner, repo, source string) []Found {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var ggufs, extras []string
	hasConfig, hasSafetensors := false, false
	for _, e := range entries {
		n := e.Name()
		switch {
		case strings.HasPrefix(n, "mmproj"):
			extras = append(extras, filepath.Join(dir, n))
		case strings.HasSuffix(n, ".gguf"):
			ggufs = append(ggufs, filepath.Join(dir, n))
		case n == "config.json":
			hasConfig = true
		case strings.HasSuffix(n, ".safetensors"):
			hasSafetensors = true
		}
	}
	var out []Found
	for _, g := range firstShards(ggufs) {
		out = append(out, Found{Path: g, Names: []string{owner + "/" + repo}, Owner: owner, Source: source, Extras: extras})
	}
	if hasConfig && hasSafetensors {
		out = append(out, Found{Path: dir, Names: []string{owner + "/" + repo}, Owner: owner, Source: source, IsDir: true})
	}
	return out
}

// firstShards drops every shard but the first so a sharded model is one entry.
func firstShards(paths []string) []string {
	var out []string
	for _, p := range paths {
		b := filepath.Base(p)
		if i := strings.LastIndex(b, "-of-"); i > 0 && strings.Contains(b[:i], "-0000") && !strings.HasSuffix(b[:i], "-00001") {
			continue
		}
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// HFCache scans the Hugging Face hub cache: models--<owner>--<repo>/snapshots/<rev>/.
func HFCache(root string) []Found {
	var out []Found
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	for _, e := range entries {
		name := e.Name()
		if !e.IsDir() || !strings.HasPrefix(name, "models--") {
			continue
		}
		parts := strings.SplitN(strings.TrimPrefix(name, "models--"), "--", 2)
		if len(parts) != 2 {
			continue
		}
		snaps, err := os.ReadDir(filepath.Join(root, name, "snapshots"))
		if err != nil {
			continue
		}
		for _, s := range snaps {
			if s.IsDir() {
				out = append(out, inRepoDir(filepath.Join(root, name, "snapshots", s.Name()), parts[0], parts[1], "hfcache")...)
			}
		}
	}
	return out
}

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
	} `json:"layers"`
}

// Ollama resolves manifests/<registry>/<namespace>/<name>/<tag> to blob files.
// Several tags commonly point at one blob (e.g. a user-created variant with
// a different num_ctx), so callers see one Found with many Names.
func Ollama(root string) ([]Found, error) {
	manifests := filepath.Join(root, "manifests")
	var out []Found
	err := filepath.WalkDir(manifests, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(manifests, p)
		if err != nil {
			return nil
		}
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) != 4 { // registry/namespace/name/tag
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		var m ollamaManifest
		if json.Unmarshal(raw, &m) != nil {
			return nil
		}
		for _, l := range m.Layers {
			if l.MediaType != "application/vnd.ollama.image.model" {
				continue
			}
			blob := filepath.Join(root, "blobs", strings.Replace(l.Digest, ":", "-", 1))
			if _, err := os.Stat(blob); err != nil {
				continue
			}
			out = append(out, Found{Path: blob, Names: []string{parts[2] + ":" + parts[3]}, Owner: parts[1], Source: "ollama"})
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	return out, nil
}

func dedupe(s []string) []string {
	out := s[:0]
	for i, v := range s {
		if i == 0 || v != s[i-1] {
			out = append(out, v)
		}
	}
	return out
}
