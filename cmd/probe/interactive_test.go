package main

import (
	"bufio"
	"bytes"
	"strings"
	"testing"

	"github.com/jraesly/reimagined-guacamole/internal/measure"
)

func TestInteractiveQuitAndMenu(t *testing.T) {
	var out bytes.Buffer
	if err := runInteractive(strings.NewReader("6\n"), &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"1) What fits on this machine", "probe fit --suggest --online --for <task>", "5) Image / video / TTS / speech", "6) Quit"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("menu missing %q:\n%s", want, out.String())
		}
	}
	// EOF on stdin ends the loop cleanly instead of spinning.
	if err := runInteractive(strings.NewReader(""), &out); err != nil {
		t.Errorf("EOF should exit cleanly: %v", err)
	}
	out.Reset()
	_ = runInteractive(strings.NewReader("9\nq\n"), &out)
	if !strings.Contains(out.String(), "pick 1-6") {
		t.Errorf("bad choice not reported:\n%s", out.String())
	}
}

func TestAskTask(t *testing.T) {
	for in, want := range map[string]string{"1\n": "agent", "coding\n": "coding", "VISION\n": "vision", "\n": "", "juggling\n": "", "tts\n": ""} {
		var out bytes.Buffer
		got, err := askTask(bufio.NewReader(strings.NewReader(in)), &out)
		if err != nil || got != want {
			t.Errorf("askTask(%q) = %q, %v; want %q", in, got, err, want)
		}
		if strings.TrimSpace(in) == "tts" && !strings.Contains(out.String(), "probe fit --for tts") {
			t.Errorf("tts should explain why it is unsupported:\n%s", out.String())
		}
	}
}

func TestAskUpstreamWithNoServers(t *testing.T) {
	oldO, oldL := measure.OllamaBase, measure.LMStudioBase
	measure.OllamaBase, measure.LMStudioBase = "http://127.0.0.1:1", "http://127.0.0.1:1"
	defer func() { measure.OllamaBase, measure.LMStudioBase = oldO, oldL }()
	var out bytes.Buffer
	got, err := askUpstream(bufio.NewReader(strings.NewReader("1\nhttp://localhost:8080/v1\n")), &out)
	if err != nil || got != "http://localhost:8080/v1" {
		t.Errorf("askUpstream = %q, %v", got, err)
	}
	if !strings.Contains(out.String(), "neither Ollama nor LM Studio answered") {
		t.Errorf("missing hint:\n%s", out.String())
	}
	got, _ = askUpstream(bufio.NewReader(strings.NewReader("http://x:1/v1\n")), &out)
	if got != "http://x:1/v1" {
		t.Errorf("direct URL = %q", got)
	}
}
