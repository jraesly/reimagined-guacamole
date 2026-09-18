package suggest

import (
	"strings"
	"testing"
	"time"
)

func TestEmbeddedListIsValidAndFresh(t *testing.T) {
	l, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	// This is deliberate: the build goes red when the curation is stale.
	if err := l.CheckFresh(time.Now()); err != nil {
		t.Fatal(err)
	}
}

func TestCheckFreshRejectsOldList(t *testing.T) {
	l := &List{Updated: "2026-01-01"}
	if err := l.CheckFresh(time.Date(2026, 4, 2, 0, 0, 0, 0, time.UTC)); err == nil || !strings.Contains(err.Error(), "days old") {
		t.Errorf("err = %v", err)
	}
	if err := l.CheckFresh(time.Date(2026, 3, 31, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Errorf("89 days should be fresh: %v", err)
	}
}

func TestForMemoryPicksClass(t *testing.T) {
	l, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	cases := map[float64]string{8: "<= 16 GB", 16: "<= 16 GB", 24: "24-32 GB", 32: "24-32 GB", 64: "48-64 GB", 128: "96-128 GB", 512: "96-128 GB"}
	for mem, want := range cases {
		c, ok := l.ForMemory(mem)
		if !ok || c.Name != want {
			t.Errorf("ForMemory(%v) = %q, want %q", mem, c.Name, want)
		}
	}
}

func TestFilterByTask(t *testing.T) {
	l, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	c, _ := l.ForMemory(32)
	coding, ok, why := c.Filter("coding")
	if !ok || why != "" || len(coding.Models) == 0 || len(coding.Models) >= len(c.Models) {
		t.Errorf("coding filter: ok=%v why=%q kept %d of %d", ok, why, len(coding.Models), len(c.Models))
	}
	for _, e := range coding.Models {
		found := false
		for _, t := range e.Tasks {
			found = found || t == "coding"
		}
		if !found {
			t.Errorf("%s kept without the coding task", e.Name)
		}
	}
	if all, ok, _ := c.Filter(""); !ok || len(all.Models) != len(c.Models) {
		t.Error("empty task should keep everything")
	}
	if _, ok, why := c.Filter("TTS"); ok || !strings.Contains(why, "text-to-speech") {
		t.Errorf("tts: ok=%v why=%q", ok, why)
	}
	if _, ok, why := c.Filter("juggling"); ok || !strings.Contains(why, "known tasks: agent, chat, coding, vision") {
		t.Errorf("unknown: ok=%v why=%q", ok, why)
	}
}

func TestValidateRejectsBadLists(t *testing.T) {
	ok := Entry{Name: "x", Shape: "s", Why: "w", Pull: "p", Host: "ollama", Ref: "ollama:x", Tasks: []string{"coding"}}
	bad := []List{
		{Updated: "yesterday"},
		{Updated: "2026-01-01"},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{ok}}, {Name: "b", MaxGB: 16, Models: []Entry{ok}}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{{Name: "x"}}}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{{Name: "x", Shape: "s", Why: "w", Pull: "p", Host: "s3", Ref: "s3:x"}}}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{{Name: "x", Shape: "s", Why: "w", Pull: "p", Host: "hf", Ref: "ollama:x"}}}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{{Name: "x", Shape: "s", Why: "w", Pull: "p", Host: "hf", Ref: "hf:x/y"}}}}},
		{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{{Name: "x", Shape: "s", Why: "w", Pull: "p", Host: "hf", Ref: "hf:x/y", Tasks: []string{"juggling"}}}}}},
	}
	good := List{Updated: "2026-01-01", Classes: []Class{{Name: "a", MaxGB: 32, Models: []Entry{ok}}}}
	if err := good.validate(); err != nil {
		t.Errorf("valid list rejected: %v", err)
	}
	for i, l := range bad {
		if err := l.validate(); err == nil {
			t.Errorf("case %d should fail validation", i)
		}
	}
}
