package calib

import (
	"path/filepath"
	"testing"
	"time"
)

func TestLoadMissingIsEmpty(t *testing.T) {
	f, err := Load(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil || f.Version != 1 || len(f.Samples) != 0 {
		t.Fatalf("f=%+v err=%v", f, err)
	}
}

func TestSaveLoadAddReplaceLookup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "calibration.json")
	f := &File{Version: 1, Chip: "Apple M1 Max"}
	f.Add(Sample{Model: "qwen3.8:27b", Kind: "dense", Backend: "ollama", TokPerSec: 10, BytesPerToken: 16e9, MeasuredAt: time.Now()})
	f.Add(Sample{Model: "qwen3.8:27b", Kind: "dense", Backend: "ollama", TokPerSec: 11.35, BytesPerToken: 16e9, MeasuredAt: time.Now()})
	f.Add(Sample{Model: "qwen3.6:35b", Kind: "moe", Backend: "ollama", TokPerSec: 23.5, BytesPerToken: 2.1e9, MeasuredAt: time.Now()})
	if len(f.Samples) != 2 {
		t.Fatalf("Add should replace same model+backend, have %d", len(f.Samples))
	}
	if err := f.Save(path); err != nil {
		t.Fatal(err)
	}
	g, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	s, ok := g.Lookup("other", "qwen3.8:27b")
	if !ok || s.TokPerSec != 11.35 {
		t.Errorf("lookup = %+v %v", s, ok)
	}
	if _, ok := g.Lookup("missing"); ok {
		t.Error("missing lookup should fail")
	}
	if eff, n := g.Effective("dense", ""); n != 1 || eff != 16*11.35 {
		t.Errorf("dense effective = %v (%d)", eff, n)
	}
	if eff, n := g.Effective("moe", ""); n != 1 || eff < 49 || eff > 50 {
		t.Errorf("moe effective = %v (%d)", eff, n)
	}
	if _, n := g.Effective("other", ""); n != 0 {
		t.Error("unknown kind should have no samples")
	}
	// Same-backend samples win; an unknown backend falls back to all.
	g.Add(Sample{Model: "lm", Kind: "dense", Backend: "lmstudio", TokPerSec: 10, BytesPerToken: 16e9})
	if eff, n := g.Effective("dense", "lmstudio"); n != 1 || eff != 160 {
		t.Errorf("lmstudio-only effective = %v (%d)", eff, n)
	}
	if eff, n := g.Effective("dense", "ollama"); n != 1 || eff != 16*11.35 {
		t.Errorf("ollama-only effective = %v (%d)", eff, n)
	}
	if _, n := g.Effective("dense", "vllm"); n != 2 {
		t.Errorf("unknown backend should use every sample, got %d", n)
	}
	// Aliases resolve to the same sample and never create a second one.
	g.Add(Sample{Model: "qwen3.8:27b-32k", Aliases: []string{"qwen3.8:27b"}, Kind: "dense", Backend: "ollama", TokPerSec: 13.1, BytesPerToken: 16e9})
	if len(g.Samples) != 3 { // qwen3.8 (replaced), qwen3.6, lm
		t.Errorf("alias overlap should replace, have %d samples", len(g.Samples))
	}
	if s, ok := g.Lookup("qwen3.8:27b"); !ok || s.TokPerSec != 13.1 {
		t.Errorf("lookup by alias = %+v %v", s, ok)
	}
	if eff, n := g.Effective("dense", "ollama"); n != 1 || eff != 16*13.1 {
		t.Errorf("one run under two tags must count once: %v (%d)", eff, n)
	}
}

func TestEffectiveMedian(t *testing.T) {
	f := &File{}
	for i, tps := range []float64{10, 30, 20, 40} {
		f.Add(Sample{Model: string(rune('a' + i)), Kind: "dense", TokPerSec: tps, BytesPerToken: 1e9})
	}
	if eff, n := f.Effective("dense", ""); n != 4 || eff != 25 {
		t.Errorf("median = %v (%d), want 25", eff, n)
	}
}

func TestLoadCorrupt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.json")
	f := &File{Version: 1}
	if err := f.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := writeFile(path, "{not json"); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("corrupt file should error")
	}
}
