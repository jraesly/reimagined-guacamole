package hw

import (
	"math"
	"strings"
	"testing"
)

func TestParseDarwinM1Max(t *testing.T) {
	in, err := ParseDarwin("Apple M1 Max\n", "34359738368\n", "0\n")
	if err != nil {
		t.Fatal(err)
	}
	if in.RAMGB != 32 || !in.Unified || in.WiredLimitGB != 0 {
		t.Errorf("info = %+v", in)
	}
	if !in.BandwidthKnown || in.BandwidthGBs != 400 {
		t.Errorf("bandwidth = %v known=%v", in.BandwidthGBs, in.BandwidthKnown)
	}
	b := in.Budget(DefaultReserveGB(in))
	if b.Binding != "reserve" || b.GB != 24 {
		t.Errorf("budget = %+v", b)
	}
}

func TestParseDarwinWiredLimitBinds(t *testing.T) {
	in, _ := ParseDarwin("Apple M1 Max", "34359738368", "20480")
	b := in.Budget(8)
	if b.Binding != "wired-limit" || b.GB != 20 || !strings.Contains(b.Reason, "iogpu.wired_limit_mb") {
		t.Errorf("budget = %+v", b)
	}
}

func TestParseDarwinDefaultWiredLimitBindsOnBigMachines(t *testing.T) {
	// 128 GB: reserve leaves 120, but the default ~75% wired limit is 96.
	in, _ := ParseDarwin("Apple M5 Max", "137438953472", "0")
	b := in.Budget(8)
	if b.Binding != "wired-limit" || math.Abs(b.GB-96) > 0.01 {
		t.Errorf("budget = %+v", b)
	}
	if in.BandwidthKnown || len(in.Notes) == 0 {
		t.Errorf("M5 Max should be unknown bandwidth with a note: %+v", in)
	}
}

func TestParseDarwinIntelIsNotUnified(t *testing.T) {
	in, err := ParseDarwin("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz", "68719476736", "0")
	if err != nil {
		t.Fatal(err)
	}
	if in.Unified || in.BandwidthKnown || len(in.Notes) == 0 {
		t.Errorf("info = %+v", in)
	}
	if b := in.Budget(DefaultReserveGB(in)); b.Binding != "reserve" || b.GB != 62 {
		t.Errorf("budget = %+v", b)
	}
}

func TestParseDarwinBadMemsize(t *testing.T) {
	if _, err := ParseDarwin("Apple M1", "lots", "0"); err == nil {
		t.Error("expected error")
	}
}

func TestParseLinux4090(t *testing.T) {
	meminfo := "MemTotal:       65536000 kB\nMemFree:        1000 kB\n"
	in, err := ParseLinux(meminfo, "NVIDIA GeForce RTX 4090, 24564\n")
	if err != nil {
		t.Fatal(err)
	}
	if in.Unified || in.Chip != "NVIDIA GeForce RTX 4090" || math.Abs(in.VRAMGB-23.99) > 0.01 {
		t.Errorf("info = %+v", in)
	}
	if in.BandwidthGBs != 1008 {
		t.Errorf("bandwidth = %v", in.BandwidthGBs)
	}
	b := in.Budget(DefaultReserveGB(in))
	if b.Binding != "vram" || math.Abs(b.GB-21.99) > 0.01 {
		t.Errorf("budget = %+v", b)
	}
}

func TestParseLinuxNoGPU(t *testing.T) {
	in, err := ParseLinux("MemTotal: 16777216 kB\n", "")
	if err != nil {
		t.Fatal(err)
	}
	if in.Chip != "cpu" || in.VRAMGB != 0 || len(in.Notes) == 0 {
		t.Errorf("info = %+v", in)
	}
	if b := in.Budget(2); b.Binding != "reserve" || b.GB != 14 {
		t.Errorf("budget = %+v", b)
	}
}

func TestParseLinuxMultiGPUNote(t *testing.T) {
	in, err := ParseLinux("MemTotal: 16777216 kB\n", "NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 3090, 24576\n")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, n := range in.Notes {
		found = found || strings.Contains(n, "multiple GPUs")
	}
	if !found {
		t.Errorf("expected multi-GPU note, got %v", in.Notes)
	}
}

func TestParseLinuxErrors(t *testing.T) {
	if _, err := ParseLinux("MemFree: 1 kB\n", ""); err == nil {
		t.Error("missing MemTotal should error")
	}
	if _, err := ParseLinux("MemTotal: 1 kB\n", "garbage line without comma"); err == nil {
		t.Error("malformed nvidia-smi should error")
	}
}

func TestBandwidthSpecificityOrder(t *testing.T) {
	cases := map[string]float64{
		"Apple M1 Ultra": 800, "Apple M1 Max": 400, "Apple M1 Pro": 200, "Apple M1": 68,
		"Apple M4 Max": 546, "NVIDIA GeForce RTX 4090": 1008, "NVIDIA RTX 6000 Ada Generation": 960,
	}
	for chip, want := range cases {
		if got, ok := Bandwidth(chip); !ok || got != want {
			t.Errorf("Bandwidth(%q) = %v,%v want %v", chip, got, ok, want)
		}
	}
	if _, ok := Bandwidth("Intel Core i9"); ok {
		t.Error("unknown chip should be !ok")
	}
}
