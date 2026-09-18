// Package remote evaluates models that are not on disk yet by reading only
// their headers from the registry that would serve them. Ollama's registry
// and Hugging Face both honour HTTP range requests, so a GGUF header costs a
// few megabytes rather than the whole file.
package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// Endpoints are variables so tests can point them at httptest servers.
var (
	OllamaRegistry = "https://registry.ollama.ai"
	HFBase         = "https://huggingface.co"
	Client         = &http.Client{Timeout: 60 * time.Second}
	UserAgent      = "probe/0.1 (+https://github.com/jraesly/reimagined-guacamole)"

	// InitialRange is the first byte range requested for a header; it doubles
	// until the header parses, up to MaxRange.
	InitialRange = 1 << 20
	MaxRange     = 64 << 20
)

// Ref names a model on a host without downloading it.
type Ref struct {
	Host  string // "ollama" or "hf"
	Owner string // ollama namespace (default "library") or HF org/user
	Name  string // ollama model name or HF repo
	Tag   string // ollama tag (default "latest") or HF file-name filter
}

// String renders the canonical form: ollama:owner/name:tag or hf:owner/repo:filter.
func (r Ref) String() string {
	switch r.Host {
	case "ollama":
		s := "ollama:"
		if r.Owner != "library" {
			s += r.Owner + "/"
		}
		return s + r.Name + ":" + r.Tag
	case "hf":
		s := "hf:" + r.Owner + "/" + r.Name
		if r.Tag != "" {
			s += ":" + r.Tag
		}
		return s
	}
	return r.Host + ":" + r.Name
}

// ParseRef recognises "ollama:name[:tag]", "ollama:user/name[:tag]" and
// "hf:owner/repo[:file-filter]". Anything else is a local path.
func ParseRef(s string) (Ref, bool) {
	host, rest, ok := strings.Cut(s, ":")
	if !ok {
		return Ref{}, false
	}
	switch host {
	case "ollama":
		r := Ref{Host: "ollama", Owner: "library", Tag: "latest"}
		if name, tag, ok := strings.Cut(rest, ":"); ok {
			rest, r.Tag = name, tag
		}
		if owner, name, ok := strings.Cut(rest, "/"); ok {
			r.Owner, r.Name = owner, name
		} else {
			r.Name = rest
		}
		return r, r.Name != "" && r.Tag != ""
	case "hf":
		r := Ref{Host: "hf"}
		if repo, filter, ok := strings.Cut(rest, ":"); ok {
			rest, r.Tag = repo, filter
		}
		owner, name, ok := strings.Cut(rest, "/")
		if !ok || owner == "" || name == "" {
			return Ref{}, false
		}
		r.Owner, r.Name = owner, name
		return r, true
	}
	return Ref{}, false
}

// Resolve fetches enough of the referenced model to build a Model.
func Resolve(ctx context.Context, r Ref) (*model.Model, error) {
	switch r.Host {
	case "ollama":
		return resolveOllama(ctx, r)
	case "hf":
		return resolveHF(ctx, r)
	}
	return nil, fmt.Errorf("unknown host %q", r.Host)
}

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      uint64 `json:"size"`
	} `json:"layers"`
}

func resolveOllama(ctx context.Context, r Ref) (*model.Model, error) {
	base := fmt.Sprintf("%s/v2/%s/%s", OllamaRegistry, r.Owner, r.Name)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/manifests/"+r.Tag, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", UserAgent)
	resp, err := Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", r, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("%s: not found in the Ollama registry", r)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: manifest HTTP %d", r, resp.StatusCode)
	}
	var m ollamaManifest
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&m); err != nil {
		return nil, fmt.Errorf("%s: manifest: %w", r, err)
	}
	var digest string
	var size, projector uint64
	for _, l := range m.Layers {
		switch l.MediaType {
		case "application/vnd.ollama.image.model":
			digest, size = l.Digest, l.Size
		case "application/vnd.ollama.image.projector":
			projector = l.Size
		}
	}
	if digest == "" {
		return nil, fmt.Errorf("%s: manifest has no model layer (cloud-only model?)", r)
	}
	h, err := FetchHeader(ctx, base+"/blobs/"+digest)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", r, err)
	}
	name := r.Name + ":" + r.Tag
	if r.Owner != "library" {
		name = r.Owner + "/" + name
	}
	mdl, err := model.FromHeader(h, r.String(), name, size, r.Owner, false)
	if err != nil {
		return nil, err
	}
	if projector > 0 {
		mdl.Warnings = append(mdl.Warnings, fmt.Sprintf("pull also includes a %.1f GB vision projector", float64(projector)/1e9))
	}
	return mdl, nil
}

type hfEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size uint64 `json:"size"`
}

func resolveHF(ctx context.Context, r Ref) (*model.Model, error) {
	repo := r.Owner + "/" + r.Name
	body, err := get(ctx, fmt.Sprintf("%s/api/models/%s/tree/main", HFBase, repo), 4<<20)
	if err != nil {
		return nil, fmt.Errorf("hf:%s: %w", repo, err)
	}
	var entries []hfEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		return nil, fmt.Errorf("hf:%s: file listing: %w", repo, err)
	}
	var ggufs []hfEntry
	var safetensors uint64
	hasConfig := false
	for _, e := range entries {
		switch {
		case e.Type != "file":
		case strings.HasSuffix(e.Path, ".gguf") && !strings.Contains(e.Path, "mmproj"):
			ggufs = append(ggufs, e)
		case e.Path == "config.json":
			hasConfig = true
		case strings.HasSuffix(e.Path, ".safetensors"):
			safetensors += e.Size
		}
	}
	if len(ggufs) > 0 {
		chosen, err := chooseGGUF(ggufs, r.Tag)
		if err != nil {
			return nil, fmt.Errorf("hf:%s: %w", repo, err)
		}
		first, total, partial := shardGroup(chosen, ggufs)
		h, err := FetchHeader(ctx, fmt.Sprintf("%s/%s/resolve/main/%s", HFBase, repo, first.Path))
		if err != nil {
			return nil, fmt.Errorf("hf:%s: %w", repo, err)
		}
		return model.FromHeader(h, r.String(), repo+"/"+first.Path, total, r.Owner, partial)
	}
	if hasConfig && safetensors > 0 {
		raw, err := get(ctx, fmt.Sprintf("%s/%s/resolve/main/config.json", HFBase, repo), 1<<20)
		if err != nil {
			return nil, fmt.Errorf("hf:%s: %w", repo, err)
		}
		mdl, err := model.FromMLXConfig(raw, r.String(), repo, safetensors, r.Owner)
		if err != nil {
			return nil, err
		}
		mdl.Remote = true
		return mdl, nil
	}
	return nil, fmt.Errorf("hf:%s: no .gguf files and no config.json + safetensors", repo)
}

// preferredQuants is the order tried when the user gives no file filter.
var preferredQuants = []string{"Q4_K_M", "Q4_K_XL", "Q4_K_S", "IQ4_XS", "Q4_0", "Q5_K_M", "Q8_0"}

func chooseGGUF(files []hfEntry, filter string) (hfEntry, error) {
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	if filter != "" {
		for _, f := range files {
			if strings.Contains(strings.ToLower(f.Path), strings.ToLower(filter)) {
				return f, nil
			}
		}
		names := make([]string, 0, len(files))
		for _, f := range files {
			names = append(names, f.Path)
		}
		return hfEntry{}, fmt.Errorf("no .gguf matches %q; available: %s", filter, strings.Join(names, ", "))
	}
	for _, q := range preferredQuants {
		for _, f := range files {
			if strings.Contains(strings.ToUpper(f.Path), q) {
				return f, nil
			}
		}
	}
	return files[0], nil
}

// shardGroup returns the first shard of chosen's group, the group's total
// size, and whether chosen is part of a multi-file group.
func shardGroup(chosen hfEntry, all []hfEntry) (hfEntry, uint64, bool) {
	i := strings.LastIndex(chosen.Path, "-of-")
	if i < 0 || !strings.Contains(chosen.Path[:i], "-0000") {
		return chosen, chosen.Size, false
	}
	prefix := chosen.Path[:strings.LastIndex(chosen.Path[:i], "-")]
	suffix := chosen.Path[i:]
	first, total := chosen, uint64(0)
	for _, f := range all {
		if strings.HasPrefix(f.Path, prefix+"-") && strings.HasSuffix(f.Path, suffix) {
			total += f.Size
			if f.Path < first.Path {
				first = f
			}
		}
	}
	return first, total, true
}

// FetchHeader reads a GGUF header from url using growing range requests.
func FetchHeader(ctx context.Context, url string) (*gguf.Header, error) {
	for n := InitialRange; n <= MaxRange; n *= 2 {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", n-1))
		req.Header.Set("User-Agent", UserAgent)
		resp, err := Client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("header fetch HTTP %d", resp.StatusCode)
		}
		// A server that ignores Range answers 200 with the whole file; read
		// only what this round would have asked for.
		data, err := io.ReadAll(io.LimitReader(resp.Body, int64(n)))
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		h, perr := gguf.Parse(bytes.NewReader(data))
		if perr == nil {
			return h, nil
		}
		if !strings.Contains(perr.Error(), "truncated") || len(data) < n {
			return nil, perr
		}
	}
	return nil, errors.New("header larger than the range limit")
}

func get(ctx context.Context, url string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", UserAgent)
	resp, err := Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, errors.New("not found")
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, limit))
}
