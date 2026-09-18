package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jraesly/reimagined-guacamole/internal/suggest"
)

func TestDetectHostsAndFilter(t *testing.T) {
	home := t.TempDir()
	t.Setenv("PATH", home) // no ollama binary on PATH
	if hosts := detectHosts(home); len(hosts) != 0 {
		t.Errorf("empty home should detect nothing, got %v", hosts)
	}
	if err := os.MkdirAll(filepath.Join(home, ".lmstudio", "models"), 0o755); err != nil {
		t.Fatal(err)
	}
	hosts := detectHosts(home)
	if !hosts["hf"] || hosts["ollama"] {
		t.Errorf("hosts = %v, want hf only", hosts)
	}
	entries := []suggest.Entry{{Name: "a", Host: "ollama"}, {Name: "b", Host: "hf"}}
	if got := filterHosts(entries, hosts); len(got) != 1 || got[0].Name != "b" {
		t.Errorf("filtered = %v", got)
	}
	if got := filterHosts(entries, nil); len(got) != 2 {
		t.Errorf("no hosts should keep all, got %v", got)
	}
}

func TestParseContexts(t *testing.T) {
	cases := []struct {
		in   string
		want []uint64
		err  bool
	}{
		{"", nil, false},
		{"32k", []uint64{32768}, false},
		{"8k, 32K ,131072", []uint64{8192, 32768, 131072}, false},
		{"0", nil, true},
		{"lots", nil, true},
		{"32kb", nil, true},
	}
	for _, c := range cases {
		got, err := parseContexts(c.in)
		if (err != nil) != c.err {
			t.Errorf("parseContexts(%q) err = %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseContexts(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}
