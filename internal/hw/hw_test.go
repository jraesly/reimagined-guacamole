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

func TestBandwidthAMD(t *testing.T) {
	for _, tc := range []struct {
		chip string
		want float64
	}{
		{"Navi 31 [Radeon RX 7900 XTX]", 960}, {"radeon rx 7900 xt", 800},
		{"RX 7900 GRE", 576}, {"RX 7800 XT", 624}, {"RX 7700 XT", 432},
		{"RX 6950 XT", 576}, {"RX 6900 XT", 512}, {"RX 6800 XT", 512},
		{"Radeon PRO W7900", 864}, {"Radeon PRO W7800", 576},
		{"AMD Instinct MI300X", 5300}, {"MI250X", 3277}, {"MI210", 1638}, {"MI100", 1229},
	} {
		t.Run(tc.chip, func(t *testing.T) {
			if got, known := Bandwidth(tc.chip); !known || got != tc.want {
				t.Fatalf("got %v, %v; want %v", got, known, tc.want)
			}
		})
	}
}

func TestParseAMD(t *testing.T) {
	const rocm = `{"card0":{"VRAM Total Memory (B)":"25753026560","VRAM Total Used Memory (B)":"123","Card Series":"Navi 31 [Radeon RX 7900 XTX]","Card Model":"0x744c"}}`
	fallback := []SysfsCard{{Card: "card0", VRAMTotalBytes: "17179869184\n", Vendor: "0x1002\n"}}
	for _, tc := range []struct {
		name, raw       string
		cards           []SysfsCard
		chip            string
		vram, bandwidth float64
		note            string
	}{
		{"rocm", rocm, fallback, "Navi 31 [Radeon RX 7900 XTX]", 23.984375, 960, "ROCm (HIP)"},
		{"sysfs_unknown", "", fallback, "AMD GPU (card0)", 16, 0, "not in the table"},
		{"malformed_fallback", "{broken", fallback, "AMD GPU (card0)", 16, 0, "ROCm (HIP)"},
		{"version_keys", `{"card0":{"vram total memory (bytes)":25769803776,"DEVICE NAME":"RX 7900 XTX"}}`, nil, "RX 7900 XTX", 24, 960, "full offload"},
		{"model_key", `{"card0":{"VRAM Total Memory":"17179869184","card model":"RX 6800 XT"}}`, nil, "RX 6800 XT", 16, 512, "full offload"},
		{"multi_rocm", `{"card1":{"VRAM Total Memory (B)":"17179869184","Card Series":"RX 6800 XT"},"card0":{"VRAM Total Memory (B)":"25769803776","Card Series":"RX 7900 XTX"}}`, nil, "RX 7900 XTX", 24, 960, "multiple GPUs"},
		{"multi_sysfs", "", append(append([]SysfsCard{}, fallback...), SysfsCard{Card: "card1", VRAMTotalBytes: "8589934592", Vendor: "0x1002"}), "AMD GPU (card0)", 16, 0, "multiple GPUs"},
		{"pci_vendor", "", []SysfsCard{{Card: "card0", VRAMTotalBytes: "17179869184", Vendor: "PCI_ID=1002:744C"}}, "AMD GPU (card0)", 16, 0, "full offload"},
		{"sysfs_product", "", []SysfsCard{{Card: "card0", VRAMTotalBytes: "17179869184", ProductName: "RX 6800 XT", Vendor: "0x1002"}}, "RX 6800 XT", 16, 512, "full offload"},
		{"invalid_vram", "", []SysfsCard{{Card: "card0", VRAMTotalBytes: "-1", Vendor: "0x1002"}}, "", 0, 0, ""},
		{"non_amd", "", []SysfsCard{{Card: "card0", VRAMTotalBytes: "17179869184", Vendor: "0x10de"}}, "", 0, 0, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseAMD(tc.raw, tc.cards)
			if got.Chip != tc.chip || math.Abs(got.VRAMGB-tc.vram) > 0.000001 || got.BandwidthGBs != tc.bandwidth || got.BandwidthKnown != (tc.bandwidth > 0) || got.Unified {
				t.Fatalf("got %+v", got)
			}
			if tc.note != "" && !strings.Contains(strings.Join(got.Notes, " "), tc.note) {
				t.Fatalf("missing note %q: %v", tc.note, got.Notes)
			}
			if tc.vram > 0 {
				if b := got.Budget(2); b.GB != tc.vram-2 || b.Binding != "vram" {
					t.Fatalf("budget: %+v", b)
				}
			}
		})
	}
}

func TestParseLinuxAMDPrecedence(t *testing.T) {
	for _, tc := range []struct {
		name, nvidia, rocm, chip string
		vram                     float64
	}{
		{"nvidia", "NVIDIA GeForce RTX 4090, 24576", `{"card0":{"VRAM Total Memory":"17179869184","Card Series":"RX 6800 XT"}}`, "NVIDIA GeForce RTX 4090", 24},
		{"rocm", "", `{"card0":{"VRAM Total Memory":"17179869184","Card Series":"RX 6800 XT"}}`, "RX 6800 XT", 16},
		{"sysfs", "", "bad json", "AMD GPU (card0)", 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseLinuxWithAMD("MemTotal: 33554432 kB", tc.nvidia, tc.rocm, []SysfsCard{{Card: "card0", VRAMTotalBytes: "8589934592", Vendor: "0x1002"}})
			if err != nil {
				t.Fatal(err)
			}
			if got.Chip != tc.chip || got.VRAMGB != tc.vram || got.RAMGB != 32 || got.OS != "linux" {
				t.Fatalf("got %+v", got)
			}
			if strings.Contains(strings.Join(got.Notes, " "), "CPU-only") {
				t.Fatal(got.Notes)
			}
		})
	}
}
