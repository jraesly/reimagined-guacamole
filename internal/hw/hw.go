// Package hw detects the memory and bandwidth facts fit needs. Detection is
// split into thin OS calls and pure parsing functions so the parsing can be
// tested without the hardware.
package hw

import (
	"bufio"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

const GiB = 1024 * 1024 * 1024

// Info is what fit knows about the machine.
type Info struct {
	OS             string
	Chip           string
	RAMGB          float64
	VRAMGB         float64 // discrete GPU memory; 0 on unified-memory machines
	Unified        bool
	WiredLimitGB   float64 // macOS iogpu.wired_limit_mb if set; 0 means default
	BandwidthGBs   float64
	BandwidthKnown bool
	Notes          []string
}

// Detect inspects the current machine.
func Detect() (Info, error) {
	switch runtime.GOOS {
	case "darwin":
		brand, _ := sysctl("machdep.cpu.brand_string")
		mem, _ := sysctl("hw.memsize")
		wired, _ := sysctl("iogpu.wired_limit_mb")
		return ParseDarwin(brand, mem, wired)
	case "linux":
		meminfo, err := runOut("cat", "/proc/meminfo")
		if err != nil {
			return Info{}, err
		}
		smi, _ := runOut("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
		rocm, _ := runOut("rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json")
		return ParseLinuxWithAMD(meminfo, smi, rocm, readSysfsCards())
	}
	return Info{}, fmt.Errorf("unsupported OS %s", runtime.GOOS)
}

// ParseDarwin builds Info from sysctl strings.
func ParseDarwin(brand, memsize, wiredMB string) (Info, error) {
	mem, err := strconv.ParseUint(strings.TrimSpace(memsize), 10, 64)
	if err != nil {
		return Info{}, fmt.Errorf("hw.memsize: %w", err)
	}
	in := Info{OS: "darwin", Chip: strings.TrimSpace(brand), RAMGB: float64(mem) / GiB}
	in.Unified = strings.Contains(in.Chip, "Apple")
	if !in.Unified {
		in.Notes = append(in.Notes, "Intel Mac: GPU memory is not detected; the budget assumes CPU inference from system RAM")
	}
	if w, err := strconv.ParseFloat(strings.TrimSpace(wiredMB), 64); err == nil && w > 0 && in.Unified {
		in.WiredLimitGB = w / 1024
	}
	in.BandwidthGBs, in.BandwidthKnown = Bandwidth(in.Chip)
	if !in.BandwidthKnown {
		in.Notes = append(in.Notes, "memory bandwidth for "+in.Chip+" is not in the table; pass --bandwidth-gb-s to get a speed estimate")
	}
	return in, nil
}

// ParseLinux builds Info from /proc/meminfo and optional nvidia-smi output.
func ParseLinux(meminfo, nvidiaSMI string) (Info, error) {
	in := Info{OS: "linux"}
	sc := bufio.NewScanner(strings.NewReader(meminfo))
	for sc.Scan() {
		if f := strings.Fields(sc.Text()); len(f) >= 2 && f[0] == "MemTotal:" {
			kb, err := strconv.ParseFloat(f[1], 64)
			if err != nil {
				return Info{}, fmt.Errorf("MemTotal: %w", err)
			}
			in.RAMGB = kb * 1024 / GiB
		}
	}
	if in.RAMGB == 0 {
		return Info{}, errors.New("MemTotal not found in /proc/meminfo")
	}
	line := strings.TrimSpace(strings.Split(strings.TrimSpace(nvidiaSMI), "\n")[0])
	if line == "" {
		in.Notes = append(in.Notes, "no supported GPU detected; CPU-only fit uses system RAM")
		in.Chip = "cpu"
		return in, nil
	}
	parts := strings.Split(line, ",")
	if len(parts) != 2 {
		return Info{}, fmt.Errorf("unexpected nvidia-smi line %q", line)
	}
	in.Chip = strings.TrimSpace(parts[0])
	mib, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return Info{}, fmt.Errorf("nvidia-smi memory.total: %w", err)
	}
	in.VRAMGB = mib / 1024
	in.BandwidthGBs, in.BandwidthKnown = Bandwidth(in.Chip)
	if !in.BandwidthKnown {
		in.Notes = append(in.Notes, "memory bandwidth for "+in.Chip+" is not in the table; pass --bandwidth-gb-s to get a speed estimate")
	}
	if strings.Count(strings.TrimSpace(nvidiaSMI), "\n") > 0 {
		in.Notes = append(in.Notes, "multiple GPUs found; fit uses the first one only")
	}
	return in, nil
}

// Budget is the memory available for weights + KV cache and why.
type Budget struct {
	GB      float64
	Reason  string
	Binding string // "wired-limit", "reserve", or "vram"
}

// DefaultReserveGB is what fit keeps for the OS, harness and browser.
func DefaultReserveGB(in Info) float64 {
	if in.Unified {
		return 8
	}
	return 2
}

// Budget computes the usable memory. On unified memory the GPU cannot wire
// more than macOS allows (iogpu.wired_limit_mb, or about 75% of RAM by
// default), so the budget is the smaller of that and RAM minus reserve.
func (in Info) Budget(reserveGB float64) Budget {
	if !in.Unified {
		if in.VRAMGB > 0 {
			return Budget{GB: in.VRAMGB - reserveGB, Binding: "vram",
				Reason: fmt.Sprintf("%.1f GB VRAM minus %.0f GB reserve", in.VRAMGB, reserveGB)}
		}
		return Budget{GB: in.RAMGB - reserveGB, Binding: "reserve",
			Reason: fmt.Sprintf("%.1f GB RAM minus %.0f GB reserve (no GPU)", in.RAMGB, reserveGB)}
	}
	byReserve := in.RAMGB - reserveGB
	wired := in.WiredLimitGB
	wiredWhy := "macOS default wired limit (~75% of RAM)"
	if wired == 0 {
		wired = in.RAMGB * 0.75
	} else {
		wiredWhy = "iogpu.wired_limit_mb"
	}
	if wired < byReserve {
		return Budget{GB: wired, Binding: "wired-limit",
			Reason: fmt.Sprintf("%.1f GB from %s; raising it would add up to %.1f GB", wired, wiredWhy, byReserve-wired)}
	}
	return Budget{GB: byReserve, Binding: "reserve",
		Reason: fmt.Sprintf("%.1f GB RAM minus %.0f GB reserve for OS, harness and browser", in.RAMGB, reserveGB)}
}

// bandwidthTable maps a chip-name fragment to memory bandwidth in GB/s.
// Order matters: more specific names come first. Values are vendor figures
// for the common bin; some chips ship in a lower-bandwidth bin.
var bandwidthTable = []struct {
	match string
	gbs   float64
}{
	{"M1 Ultra", 800}, {"M1 Max", 400}, {"M1 Pro", 200}, {"M1", 68},
	{"M2 Ultra", 800}, {"M2 Max", 400}, {"M2 Pro", 200}, {"M2", 100},
	{"M3 Ultra", 819}, {"M3 Max", 400}, {"M3 Pro", 150}, {"M3", 100},
	{"M4 Max", 546}, {"M4 Pro", 273}, {"M4", 120},
	{"RTX 5090", 1792}, {"RTX 5080", 960}, {"RTX 5070 Ti", 896}, {"RTX 5070", 672}, {"RTX 5060 Ti", 448},
	{"RTX 4090", 1008}, {"RTX 4080 Super", 736}, {"RTX 4080", 717}, {"RTX 4070 Ti Super", 672},
	{"RTX 4070 Ti", 504}, {"RTX 4070 Super", 504}, {"RTX 4070", 504}, {"RTX 4060 Ti", 288},
	{"RTX 3090 Ti", 1008}, {"RTX 3090", 936}, {"RTX 3080 Ti", 912}, {"RTX 3080", 760}, {"RTX 3070", 448},
	{"RTX 6000 Ada", 960}, {"RTX A6000", 768}, {"L40S", 864}, {"L40", 864}, {"L4", 300},
	{"A100", 1935}, {"H100", 3350}, {"H200", 4800},
	{"RX 7900 XTX", 960}, {"RX 7900 XT", 800}, {"RX 7900 GRE", 576},
	{"RX 7800 XT", 624}, {"RX 7700 XT", 432}, {"RX 6950 XT", 576},
	{"RX 6900 XT", 512}, {"RX 6800 XT", 512},
	{"W7900", 864}, {"W7800", 576},
	{"MI300X", 5300}, {"MI250X", 3277}, {"MI210", 1638}, {"MI100", 1229},
}

// Bandwidth looks up memory bandwidth by chip name.
func Bandwidth(chip string) (float64, bool) {
	c := strings.ToLower(chip)
	for _, e := range bandwidthTable {
		if strings.Contains(c, strings.ToLower(e.match)) {
			return e.gbs, true
		}
	}
	return 0, false
}

func sysctl(key string) (string, error) { return runOut("sysctl", "-n", key) }

func runOut(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	return string(out), err
}
