package hw

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SysfsCard contains raw Linux DRM device attributes. Vendor may be a vendor
// ID (0x1002), a PCI ID (1002:744c), or the contents of the uevent file.
type SysfsCard struct {
	Card           string
	VRAMTotalBytes string
	ProductName    string
	Vendor         string
}

// ParseLinuxWithAMD preserves NVIDIA precedence and uses AMD before CPU-only
// fallback. Invalid NVIDIA output retains ParseLinux error behavior.
func ParseLinuxWithAMD(meminfo, nvidiaSMI, rocmJSON string, sysfs []SysfsCard) (Info, error) {
	in, err := ParseLinux(meminfo, nvidiaSMI)
	if err != nil || strings.TrimSpace(nvidiaSMI) != "" {
		return in, err
	}
	if amd := ParseAMD(rocmJSON, sysfs); amd.VRAMGB > 0 {
		amd.OS, amd.RAMGB = in.OS, in.RAMGB
		return amd, nil
	}
	return in, nil
}

// ParseAMD uses the first usable ROCm card, falling back to sysfs when ROCm
// output is missing or invalid. A zero Info means no AMD VRAM was detected.
func ParseAMD(rocmJSON string, sysfs []SysfsCard) Info {
	var devices map[string]map[string]json.RawMessage
	var cards []SysfsCard
	if json.Unmarshal([]byte(rocmJSON), &devices) == nil {
		for card, fields := range devices {
			if !strings.HasPrefix(strings.ToLower(card), "card") {
				continue
			}
			value := func(prefixes ...string) string {
				// Sort keys to make version-specific duplicate fields deterministic.
				keys := make([]string, 0, len(fields))
				for k := range fields {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				for _, prefix := range prefixes {
					for _, k := range keys {
						if !strings.HasPrefix(strings.ToLower(k), prefix) {
							continue
						}
						var s string
						if json.Unmarshal(fields[k], &s) != nil {
							s = string(fields[k])
						}
						s = strings.TrimSpace(s)
						if s != "" && s != "null" {
							return s
						}
					}
				}
				return ""
			}
			c := SysfsCard{Card: card, VRAMTotalBytes: value("vram total memory"), ProductName: value("card series", "device name", "card model")}
			if validVRAM(c.VRAMTotalBytes) > 0 {
				cards = append(cards, c)
			}
		}
	}
	if len(cards) == 0 {
		for _, c := range sysfs {
			vendor := strings.ToLower(strings.TrimSpace(c.Vendor))
			for _, line := range strings.Split(vendor, "\n") {
				if strings.HasPrefix(line, "pci_id=") {
					vendor = strings.TrimPrefix(line, "pci_id=")
					break
				}
			}
			amd := vendor == "0x1002" || vendor == "1002" || strings.HasPrefix(vendor, "1002:")
			product := strings.ToLower(c.ProductName)
			if vendor == "" {
				amd = strings.Contains(product, "amd") || strings.Contains(product, "radeon")
			}
			if amd && validVRAM(c.VRAMTotalBytes) > 0 {
				cards = append(cards, c)
			}
		}
	}
	if len(cards) == 0 {
		return Info{}
	}
	sort.SliceStable(cards, func(i, j int) bool {
		a, ae := strconv.Atoi(strings.TrimPrefix(cards[i].Card, "card"))
		b, be := strconv.Atoi(strings.TrimPrefix(cards[j].Card, "card"))
		if ae == nil && be == nil {
			return a < b
		}
		return cards[i].Card < cards[j].Card
	})
	c := cards[0]
	in := Info{Chip: strings.TrimSpace(c.ProductName), VRAMGB: validVRAM(c.VRAMTotalBytes)}
	if in.Chip == "" {
		in.Chip = "AMD GPU (" + c.Card + ")"
	}
	in.BandwidthGBs, in.BandwidthKnown = Bandwidth(in.Chip)
	if !in.BandwidthKnown {
		in.Notes = append(in.Notes, "memory bandwidth for "+in.Chip+" is not in the table; pass --bandwidth-gb-s to get a speed estimate")
	}
	if len(cards) > 1 {
		in.Notes = append(in.Notes, "multiple GPUs found; fit uses the first one only")
	}
	in.Notes = append(in.Notes, "llama.cpp/Ollama need a ROCm (HIP) build for offload on AMD; fit assumes full offload")
	return in
}

func validVRAM(raw string) float64 {
	bytes, err := strconv.ParseUint(strings.TrimSpace(raw), 10, 64)
	if err != nil {
		return 0
	}
	return float64(bytes) / GiB
}

func readSysfsCards() []SysfsCard {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	var cards []SysfsCard
	for _, path := range paths {
		device := filepath.Dir(path)
		read := func(name string) string { b, _ := os.ReadFile(filepath.Join(device, name)); return string(b) }
		vendor := read("vendor")
		if strings.TrimSpace(vendor) == "" {
			vendor = read("uevent")
		}
		cards = append(cards, SysfsCard{Card: filepath.Base(filepath.Dir(device)), VRAMTotalBytes: read("mem_info_vram_total"), ProductName: read("product_name"), Vendor: vendor})
	}
	return cards
}
