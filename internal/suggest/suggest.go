// Package suggest holds the one piece of fit that is curation rather than
// measurement: a short, dated list of baseline models per memory class.
package suggest

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

//go:embed models.json
var raw []byte

// MaxAge is how old the list may be before probe refuses to show it.
const MaxAge = 90 * 24 * time.Hour

// Entry is one recommended model.
type Entry struct {
	Name  string   `json:"name"`
	Shape string   `json:"shape"`
	Why   string   `json:"why"`
	Pull  string   `json:"pull"`
	Host  string   `json:"host"`  // "ollama" or "hf"
	Ref   string   `json:"ref"`   // remote reference fit can resolve, e.g. ollama:qwen3.8:27b
	Tasks []string `json:"tasks"` // what the model is for; see Tasks
}

// Tasks probe can reason about: all are transformer LLM/VLM weights that
// Ollama, LM Studio or llama.cpp serve, so the fit arithmetic applies.
var Tasks = map[string]string{
	"coding": "writing and editing code through a coding agent",
	"agent":  "long tool-calling sessions; needs reliable tool use",
	"chat":   "general assistant use",
	"vision": "understanding images or screenshots alongside text",
}

// Unsupported explains tasks that have no curated suggestion list. Image,
// video, TTS and speech models on this machine are still reported by
// `probe fit --for <task>` (weights measured from safetensors/GGUF headers,
// activations not modeled); there is just no baseline list to recommend from.
var Unsupported = map[string]string{
	"tts":    "no curated text-to-speech list; `probe fit --for tts` reports the Kokoro/XTTS/MLX-audio models on this machine",
	"speech": "no curated speech-to-text list; `probe fit --for speech` reports whisper.cpp / MLX Whisper models on this machine",
	"image":  "no curated image-generation list; `probe fit --for image` reports the Stable Diffusion/Flux models on this machine (ComfyUI, Draw Things, HF cache)",
	"video":  "no curated video-generation list; `probe fit --for video` reports Wan/HunyuanVideo/LTX models on this machine",
	"music":  "music generation models are not served by any runtime probe knows; not modeled",
}

// Filter keeps entries that list task. It returns ok=false with a reason
// when the task is unknown or one probe cannot model.
func (c Class) Filter(task string) (Class, bool, string) {
	task = strings.ToLower(strings.TrimSpace(task))
	if task == "" {
		return c, true, ""
	}
	if why, bad := Unsupported[task]; bad {
		return Class{}, false, why
	}
	if _, known := Tasks[task]; !known {
		names := make([]string, 0, len(Tasks))
		for t := range Tasks {
			names = append(names, t)
		}
		sort.Strings(names)
		return Class{}, false, fmt.Sprintf("unknown task %q; known tasks: %s", task, strings.Join(names, ", "))
	}
	out := Class{Name: c.Name, MaxGB: c.MaxGB}
	for _, e := range c.Models {
		for _, t := range e.Tasks {
			if t == task {
				out.Models = append(out.Models, e)
				break
			}
		}
	}
	return out, true, ""
}

// Class groups entries by the memory they need.
type Class struct {
	Name   string  `json:"name"`
	MaxGB  float64 `json:"max_gb"`
	Models []Entry `json:"models"`
}

// List is the embedded catalog.
type List struct {
	Updated string  `json:"updated"`
	Note    string  `json:"note"`
	Classes []Class `json:"classes"`
}

// Load parses the embedded list.
func Load() (*List, error) {
	var l List
	if err := json.Unmarshal(raw, &l); err != nil {
		return nil, fmt.Errorf("embedded models.json: %w", err)
	}
	if err := l.validate(); err != nil {
		return nil, err
	}
	return &l, nil
}

func (l *List) validate() error {
	if _, err := l.UpdatedTime(); err != nil {
		return err
	}
	if len(l.Classes) == 0 {
		return fmt.Errorf("models.json has no classes")
	}
	prev := 0.0
	for _, c := range l.Classes {
		if c.MaxGB <= prev {
			return fmt.Errorf("class %q: max_gb must increase (got %v after %v)", c.Name, c.MaxGB, prev)
		}
		prev = c.MaxGB
		if len(c.Models) == 0 || len(c.Models) > 5 {
			return fmt.Errorf("class %q: must list 1-5 models, has %d", c.Name, len(c.Models))
		}
		for _, m := range c.Models {
			if m.Name == "" || m.Shape == "" || m.Why == "" || m.Pull == "" || m.Ref == "" {
				return fmt.Errorf("class %q: entry %+v is missing a field", c.Name, m)
			}
			if m.Host != "ollama" && m.Host != "hf" {
				return fmt.Errorf("class %q: entry %q has unknown host %q", c.Name, m.Name, m.Host)
			}
			if !strings.HasPrefix(m.Ref, m.Host+":") {
				return fmt.Errorf("class %q: entry %q ref %q does not match host %q", c.Name, m.Name, m.Ref, m.Host)
			}
			if len(m.Tasks) == 0 {
				return fmt.Errorf("class %q: entry %q lists no tasks", c.Name, m.Name)
			}
			for _, t := range m.Tasks {
				if _, ok := Tasks[t]; !ok {
					return fmt.Errorf("class %q: entry %q has unknown task %q", c.Name, m.Name, t)
				}
			}
		}
	}
	return nil
}

// UpdatedTime parses the list's date.
func (l *List) UpdatedTime() (time.Time, error) {
	t, err := time.Parse("2006-01-02", l.Updated)
	if err != nil {
		return time.Time{}, fmt.Errorf("models.json updated %q: %w", l.Updated, err)
	}
	return t, nil
}

// CheckFresh fails when the list is older than MaxAge at now.
func (l *List) CheckFresh(now time.Time) error {
	t, err := l.UpdatedTime()
	if err != nil {
		return err
	}
	if age := now.Sub(t); age > MaxAge {
		return fmt.Errorf("suggestion list is %d days old (limit %d); update internal/suggest/models.json", int(age.Hours()/24), int(MaxAge.Hours()/24))
	}
	return nil
}

// ForMemory picks the class whose ceiling is the smallest one at or above
// the machine's usable memory, so a 32 GB Mac gets the 24-32 GB class.
func (l *List) ForMemory(memoryGB float64) (Class, bool) {
	for _, c := range l.Classes {
		if memoryGB <= c.MaxGB {
			return c, true
		}
	}
	if n := len(l.Classes); n > 0 {
		return l.Classes[n-1], true
	}
	return Class{}, false
}
