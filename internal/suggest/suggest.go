// Package suggest holds the one piece of fit that is curation rather than
// measurement: a short, dated list of baseline models per memory class.
package suggest

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

//go:embed models.json
var raw []byte

// MaxAge is how old the list may be before probe refuses to show it.
const MaxAge = 90 * 24 * time.Hour

// Entry is one recommended model.
type Entry struct {
	Name  string `json:"name"`
	Shape string `json:"shape"`
	Why   string `json:"why"`
	Pull  string `json:"pull"`
	Host  string `json:"host"` // "ollama" or "hf"
	Ref   string `json:"ref"`  // remote reference fit can resolve, e.g. ollama:qwen3.8:27b
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
