package media

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jraesly/reimagined-guacamole/internal/gguf"
	"github.com/jraesly/reimagined-guacamole/internal/model"
)

type Kind string

const (
	Image  Kind = "image"
	Video  Kind = "video"
	TTS    Kind = "tts"
	Speech Kind = "speech"
)

type Component struct {
	Name, Path  string
	Bytes       uint64
	DType       string
	ClassName   string // config.json _class_name, when present
	BytesSource model.Source
}

type Model struct {
	Name, Path    string
	Kind          Kind
	Family        string
	Components    []Component
	WeightsBytes  uint64
	WeightsSource model.Source
	DTypeSummary  string
	Warnings      []string
}

var pipelineFamilies = map[string]string{
	"StableDiffusionPipeline": "sd15", "StableDiffusionXLPipeline": "sdxl", "StableDiffusion3Pipeline": "sd3", "FluxPipeline": "flux",
	"WanPipeline": "wan", "HunyuanVideoPipeline": "hunyuan-video", "LTXPipeline": "ltx",
}

func familyKind(family string) Kind {
	switch family {
	case "wan", "hunyuan-video", "ltx":
		return Video
	case "kokoro", "xtts", "csm", "bark", "sesame":
		return TTS
	case "whisper":
		return Speech
	}
	return Image
}

func audioFamily(dir string) string {
	var cfg struct {
		ModelType string `json:"model_type"`
		Model     string `json:"model"`
	}
	if readJSON(filepath.Join(dir, "config.json"), &cfg) == nil {
		switch cfg.ModelType {
		case "kokoro", "csm", "bark", "sesame", "whisper":
			return cfg.ModelType
		}
		if cfg.Model == "xtts" {
			return "xtts"
		}
	}
	if strings.Contains(strings.ToLower(filepath.Base(dir)), "xtts") {
		return "xtts"
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if !e.IsDir() && kokoroFile(e.Name()) {
			return "kokoro"
		}
	}
	return ""
}

func kokoroFile(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	return strings.HasPrefix(strings.ToLower(name), "kokoro-") && (ext == ".onnx" || ext == ".pth" || ext == ".safetensors")
}

func whisperMagic(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var b [4]byte
	_, err = io.ReadFull(f, b[:])
	return err == nil && binary.LittleEndian.Uint32(b[:]) == 0x67676d6c
}

func ggufFamily(h *gguf.Header) (string, bool) {
	f := h.Arch()
	switch f {
	case "flux", "sd3", "sdxl", "sd1", "wan", "hunyuan-video", "ltx", "whisper":
		return f, true
	}
	return "", false
}

// Detect is a discovery hint, not validation of a complete runnable pipeline.
// Safetensors extensions are intentionally candidates, including unknown components.
// GGUF detection reads its header to avoid classifying LLMs as diffusion models.
func Detect(path string) (Kind, bool) {
	st, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	if st.IsDir() {
		var index struct {
			ClassName string `json:"_class_name"`
		}
		if readJSON(filepath.Join(path, "model_index.json"), &index) == nil {
			return familyKind(pipelineFamilies[index.ClassName]), true
		}
		if f := audioFamily(path); f != "" {
			return familyKind(f), true
		}
		return "", false
	}
	if !st.Mode().IsRegular() {
		return "", false
	}
	name := filepath.Base(path)
	if kokoroFile(name) {
		return TTS, true
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".safetensors":
		return Image, true
	case ".gguf":
		h, err := gguf.ReadHeader(path)
		if err == nil {
			if f, ok := ggufFamily(h); ok {
				return familyKind(f), true
			}
		}
	case ".bin":
		if strings.HasPrefix(name, "ggml-") && whisperMagic(path) {
			return Speech, true
		}
	}
	return "", false
}

// Load measures stored weights; opaque formats use file size, not resident memory.
func Load(path string) (*Model, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	m := &Model{Name: filepath.Base(path), Path: path, Family: "unknown", WeightsSource: model.Measured}
	dtypes := map[string]uint64{}
	if st.IsDir() {
		if _, err := os.Stat(filepath.Join(path, "model_index.json")); err == nil {
			if err = m.loadDiffusers(dtypes); err != nil {
				return nil, err
			}
		} else {
			m.Family = audioFamily(path)
			if m.Family == "" {
				return nil, fmt.Errorf("%s: unrecognized media directory", path)
			}
			m.Kind = familyKind(m.Family)
			h, err := directoryHeader(path)
			if err != nil {
				return nil, err
			}
			if len(h.Tensors) > 0 {
				if err = m.addHeader(filepath.Base(path), path, "", h, dtypes); err != nil {
					return nil, err
				}
			} else {
				entries, err := os.ReadDir(path)
				if err != nil {
					return nil, err
				}
				for _, e := range entries {
					if e.IsDir() {
						continue
					}
					switch strings.ToLower(filepath.Ext(e.Name())) {
					case ".onnx", ".pth", ".pt", ".bin":
						p := filepath.Join(path, e.Name())
						info, err := os.Stat(p)
						if err != nil {
							return nil, err
						}
						if !info.Mode().IsRegular() {
							continue
						}
						if err = m.addOpaque(e.Name(), p, uint64(info.Size()), "unknown", dtypes); err != nil {
							return nil, err
						}
					}
				}
			}
			if len(m.Components) == 0 {
				return nil, fmt.Errorf("%s: no supported weights", path)
			}
		}
	} else {
		if !st.Mode().IsRegular() {
			return nil, fmt.Errorf("%s: not a regular file", path)
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".safetensors":
			h, err := ReadHeader(path)
			if err != nil {
				return nil, err
			}
			m.Family = checkpointFamily(h)
			m.Kind = Image
			if kokoroFile(filepath.Base(path)) {
				m.Family = "kokoro"
				m.Kind = TTS
			}
			if err = m.addHeader(m.Name, path, "", h, dtypes); err != nil {
				return nil, err
			}
		case ".gguf":
			h, err := gguf.ReadHeader(path)
			if err != nil {
				return nil, err
			}
			f, ok := ggufFamily(h)
			if !ok {
				return nil, fmt.Errorf("%s: unsupported media architecture %q", path, h.Arch())
			}
			m.Family = f
			m.Kind = familyKind(f)
			dtype := "unknown"
			if ft, ok := h.Uint("general.file_type"); ok {
				dtype = quantName(ft)
			}
			if err = m.addOpaque(m.Name, path, uint64(st.Size()), dtype, dtypes); err != nil {
				return nil, err
			}
		default:
			if kokoroFile(filepath.Base(path)) {
				m.Family = "kokoro"
				m.Kind = TTS
			} else if strings.HasPrefix(filepath.Base(path), "ggml-") && filepath.Ext(path) == ".bin" && whisperMagic(path) {
				m.Family = "whisper"
				m.Kind = Speech
			} else {
				return nil, fmt.Errorf("%s: unsupported media file or invalid magic", path)
			}
			if err = m.addOpaque(m.Name, path, uint64(st.Size()), "unknown", dtypes); err != nil {
				return nil, err
			}
		}
	}
	if m.Family == "unknown" {
		m.Warnings = append(m.Warnings, "unrecognized model family; component weights do not prove a complete pipeline")
	}
	m.DTypeSummary = dtypeSummary(dtypes)
	return m, nil
}

func (m *Model) loadDiffusers(dtypes map[string]uint64) error {
	var index map[string]json.RawMessage
	if err := readJSON(filepath.Join(m.Path, "model_index.json"), &index); err != nil {
		return err
	}
	var class string
	if err := json.Unmarshal(index["_class_name"], &class); err != nil {
		return fmt.Errorf("%s: invalid pipeline class: %w", m.Path, err)
	}
	m.Family = pipelineFamilies[class]
	if m.Family == "" {
		m.Family = "unknown"
	}
	m.Kind = familyKind(m.Family)
	keys := make([]string, 0, len(index))
	for key := range index {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		var component []json.RawMessage
		if json.Unmarshal(index[key], &component) != nil || len(component) != 2 {
			continue
		}
		if string(component[0]) == "null" && string(component[1]) == "null" {
			continue
		}
		if !filepath.IsLocal(key) || filepath.Base(key) != key {
			return fmt.Errorf("invalid component path %q", key)
		}
		dir := filepath.Join(m.Path, key)
		h, err := directoryHeader(dir)
		if err != nil {
			return fmt.Errorf("component %s: %w", key, err)
		}
		var cfg struct {
			ClassName string `json:"_class_name"`
		}
		if err = readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil && !os.IsNotExist(err) {
			return err
		}
		if len(h.Tensors) == 0 {
			m.Warnings = append(m.Warnings, "component "+key+" has no safetensors weights")
		}
		if err = m.addHeader(key, dir, cfg.ClassName, h, dtypes); err != nil {
			return err
		}
	}
	if m.WeightsBytes == 0 {
		return fmt.Errorf("%s: no safetensors weights", m.Path)
	}
	return nil
}

func (m *Model) addHeader(name, path, class string, h *Header, dtypes map[string]uint64) error {
	bytes := h.TotalBytes()
	if bytes > math.MaxUint64-m.WeightsBytes {
		return fmt.Errorf("%s: weight sum overflow", path)
	}
	types := h.ByDType()
	var names []string
	for dtype, n := range types {
		names = append(names, dtype)
		dtypes[dtype] += n
	}
	sort.Strings(names)
	m.Components = append(m.Components, Component{Name: name, Path: path, Bytes: bytes, DType: strings.Join(names, ", "), ClassName: class, BytesSource: model.Measured})
	m.WeightsBytes += bytes
	return nil
}

func (m *Model) addOpaque(name, path string, bytes uint64, dtype string, dtypes map[string]uint64) error {
	if bytes > math.MaxUint64-m.WeightsBytes {
		return fmt.Errorf("%s: weight sum overflow", path)
	}
	m.Components = append(m.Components, Component{Name: name, Path: path, Bytes: bytes, DType: dtype, BytesSource: model.Measured})
	m.WeightsBytes += bytes
	dtypes[dtype] += bytes
	m.Warnings = append(m.Warnings, name+": measured file size includes format overhead; resident memory is runtime-dependent")
	return nil
}

func checkpointFamily(h *Header) string {
	var input, conditioner, condStage, joint, double bool
	for _, t := range h.Tensors {
		double = double || strings.Contains(t.Name, "double_blocks.")
		input = input || strings.Contains(t.Name, "model.diffusion_model.input_blocks.")
		conditioner = conditioner || strings.Contains(t.Name, "conditioner.embedders")
		condStage = condStage || strings.Contains(t.Name, "cond_stage_model")
		joint = joint || strings.Contains(t.Name, "model.diffusion_model.joint_blocks")
	}
	switch {
	case double:
		return "flux"
	case input && conditioner:
		return "sdxl"
	case input && condStage:
		return "sd15"
	case joint:
		return "sd3"
	}
	return "unknown"
}

func dtypeSummary(types map[string]uint64) string {
	names := make([]string, 0, len(types))
	for name := range types {
		names = append(names, name)
	}
	sort.Strings(names)
	var parts []string
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s %.1f GB", strings.ToLower(name), float64(types[name])/(1<<30)))
	}
	return strings.Join(parts, ", ")
}

func quantName(ft uint64) string {
	names := map[uint64]string{0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}
	if name, ok := names[ft]; ok {
		return name
	}
	return fmt.Sprintf("type-%d", ft)
}
