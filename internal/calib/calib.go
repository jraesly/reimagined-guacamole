// Package calib stores decode-speed measurements taken on this machine so
// fit can estimate other models from them instead of from a chip table.
package calib

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Sample is one measured run of one model.
type Sample struct {
	Model         string    `json:"model"`             // host-facing name, e.g. ollama tag
	Aliases       []string  `json:"aliases,omitempty"` // other tags that resolve to the same weights
	Kind          string    `json:"kind"`              // "dense" or "moe"
	Backend       string    `json:"backend"`
	TokPerSec     float64   `json:"tok_per_sec"`
	PromptTokSec  float64   `json:"prompt_tok_per_sec"`
	BytesPerToken float64   `json:"bytes_per_token"` // weight bytes read per generated token
	Context       int       `json:"context"`
	OutputTokens  int       `json:"output_tokens"`
	MeasuredAt    time.Time `json:"measured_at"`
}

// EffectiveGBs is the bandwidth the run actually achieved: bytes read per
// token times tokens per second. It folds hardware and runtime efficiency
// into one number that transfers to other models of the same kind.
func (s Sample) EffectiveGBs() float64 { return s.BytesPerToken / 1e9 * s.TokPerSec }

// File is the on-disk calibration record.
type File struct {
	Version int      `json:"version"`
	Chip    string   `json:"chip"`
	Samples []Sample `json:"samples"`
}

// DefaultPath is $XDG_CONFIG_HOME/probe/calibration.json or ~/.config/probe/calibration.json.
func DefaultPath() (string, error) {
	if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
		return filepath.Join(x, "probe", "calibration.json"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "probe", "calibration.json"), nil
}

// Load reads the file; a missing file is an empty calibration.
func Load(path string) (*File, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return &File{Version: 1}, nil
	}
	if err != nil {
		return nil, err
	}
	var f File
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &f, nil
}

// Save writes the file atomically.
func (f *File) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Add records a sample, replacing an earlier one for the same backend whose
// model or aliases overlap (the same weights under another tag).
func (f *File) Add(s Sample) {
	for i, old := range f.Samples {
		if old.Backend == s.Backend && old.matches(append([]string{s.Model}, s.Aliases...)...) {
			f.Samples[i] = s
			return
		}
	}
	f.Samples = append(f.Samples, s)
}

func (s Sample) matches(names ...string) bool {
	for _, n := range names {
		if n == s.Model {
			return true
		}
		for _, a := range s.Aliases {
			if n == a {
				return true
			}
		}
	}
	return false
}

// Lookup finds a sample by any of the given names, including aliases.
func (f *File) Lookup(names ...string) (Sample, bool) {
	for _, s := range f.Samples {
		if s.matches(names...) {
			return s, true
		}
	}
	return Sample{}, false
}

// Effective returns the median effective bandwidth over samples of a kind
// and how many samples contributed.
func (f *File) Effective(kind string) (float64, int) {
	var v []float64
	for _, s := range f.Samples {
		if s.Kind == kind && s.TokPerSec > 0 && s.BytesPerToken > 0 {
			v = append(v, s.EffectiveGBs())
		}
	}
	if len(v) == 0 {
		return 0, 0
	}
	sort.Float64s(v)
	if n := len(v); n%2 == 1 {
		return v[n/2], n
	} else {
		return (v[n/2-1] + v[n/2]) / 2, n
	}
}
