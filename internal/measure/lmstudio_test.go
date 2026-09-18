package measure

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const lmResponse = `{"usage":{"prompt_tokens":76,"completion_tokens":64},"stats":{"tokens_per_second":10.653407289399471,"time_to_first_token":1.602515,"generation_time":7.516116},"runtime":{"name":"llama.cpp-mac-arm64-apple-metal-advsimd","version":"2.41.0"}}`

func fakeLMStudio(t *testing.T, loadedContext int, status int, response string) (*[][]string, *map[string]any) {
	t.Helper()
	var calls [][]string
	var body map[string]any
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v0/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "model", "publisher": "Owner", "arch": "qwen35", "quantization": "Q4_K_S", "type": "vlm", "state": "loaded", "max_context_length": 262144, "loaded_context_length": loadedContext}}})
	})
	mux.HandleFunc("/api/v0/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if len(calls) != 1 {
			t.Error("completion before explicit load")
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		w.WriteHeader(status)
		w.Write([]byte(response))
	})
	srv := httptest.NewServer(mux)
	oldBase, oldCommand := LMStudioBase, LMSCommand
	LMStudioBase, LMSCommand = srv.URL, os.Args[0]
	t.Cleanup(func() { srv.Close(); LMStudioBase, LMSCommand = oldBase, oldCommand })
	return &calls, &body
}

func TestLMStudioModels(t *testing.T) {
	fakeLMStudio(t, 4096, 200, lmResponse)
	if !LMStudioAvailable(context.Background()) {
		t.Fatal("unavailable")
	}
	models, err := LMStudioModels(context.Background())
	want := LMStudioModel{ID: "model", Publisher: "Owner", Arch: "qwen35", Quant: "Q4_K_S", Type: "vlm", State: "loaded", MaxContext: 262144, LoadedContext: 4096}
	if err != nil || len(models) != 1 || models[0] != want {
		t.Fatalf("models = %+v, %v", models, err)
	}
}

func TestLMStudioIDAndMatch(t *testing.T) {
	if got := LMStudioID("Owner", "Model-GGUF"); got != "model" {
		t.Fatal(got)
	}
	for _, tc := range []struct {
		name   string
		models []LMStudioModel
		want   string
	}{
		{"exact", []LMStudioModel{{ID: "prefix-model", Publisher: "owner"}, {ID: "model"}}, "model"},
		{"publisher contains", []LMStudioModel{{ID: "prefix-model-q4", Publisher: "OWNER"}}, "prefix-model-q4"},
		{"embeddings", []LMStudioModel{{ID: "model", Type: "embeddings"}}, ""},
		{"no match", []LMStudioModel{{ID: "prefix-model", Publisher: "other"}}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := MatchLMStudio(tc.models, "Owner", "Model-GGUF")
			if ok != (tc.want != "") || got.ID != tc.want {
				t.Fatalf("got %+v, %v", got, ok)
			}
		})
	}
	// A catalog download: repo lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF
	// is listed by LM Studio as id "nvidia/nemotron-3-nano-4b", publisher nvidia.
	models := []LMStudioModel{
		{ID: "nvidia/nemotron-3-nano-4b", Type: "llm", Publisher: "nvidia"},
		{ID: "text-embedding-nomic-embed-text-v1.5", Type: "embeddings", Publisher: "nomic-ai"},
	}
	got, ok := MatchLMStudio(models, "lmstudio-community", "NVIDIA-Nemotron-3-Nano-4B-GGUF")
	if !ok || got.ID != "nvidia/nemotron-3-nano-4b" {
		t.Errorf("vendor-prefixed id: got %+v, %v", got, ok)
	}
}

func TestLMStudioMeasurement(t *testing.T) {
	calls, body := fakeLMStudio(t, 4096, 200, lmResponse)
	r, err := LMStudio(context.Background(), "model", 4096, 128, func(name string, args ...string) error {
		*calls = append(*calls, append([]string{name}, args...))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual((*calls)[0], []string{LMSCommand, "load", "model", "--context-length", "4096", "--gpu", "max", "-y"}) || !reflect.DeepEqual((*calls)[1], []string{LMSCommand, "unload", "--all"}) {
		t.Fatalf("calls = %v", *calls)
	}
	if (*body)["model"] != "model" || (*body)["stream"] != false || (*body)["max_tokens"] != float64(128) || (*body)["temperature"] != float64(0) {
		t.Errorf("body = %v", *body)
	}
	messages := (*body)["messages"].([]any)
	if messages[0].(map[string]any)["content"] != Prompt {
		t.Error("wrong prompt")
	}
	if r.Model != "model" || r.Context != 4096 || r.OutputTokens != 64 || r.PromptTokens != 76 || r.TokPerSec < 10.65 || r.TokPerSec > 10.66 || r.TTFTSeconds < 1.60 || r.TTFTSeconds > 1.61 || r.TotalSeconds != 7.516116 || r.Runtime != "llama.cpp-mac-arm64-apple-metal-advsimd 2.41.0" || len(r.Notes) != 0 {
		t.Errorf("result = %+v", r)
	}
}

func TestLMStudioContextMismatch(t *testing.T) {
	calls, _ := fakeLMStudio(t, 8192, 200, lmResponse)
	r, err := LMStudio(context.Background(), "model", 4096, 128, func(name string, args ...string) error { *calls = append(*calls, args); return nil })
	if err != nil || len(r.Notes) == 0 || r.Context != 8192 {
		t.Fatalf("result = %+v, %v", r, err)
	}
}

func TestLMStudioErrorsUnload(t *testing.T) {
	for _, tc := range []struct {
		name     string
		status   int
		response string
		loadErr  bool
		want     string
	}{
		{"post failure", 500, `{"error":"failed"}`, false, "HTTP 500"},
		{"missing stats", 200, `{"usage":{"completion_tokens":64}}`, false, "stats"},
		{"load failure", 200, lmResponse, true, "load"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, _ := fakeLMStudio(t, 4096, tc.status, tc.response)
			_, err := LMStudio(context.Background(), "model", 4096, 128, func(name string, args ...string) error {
				*calls = append(*calls, args)
				if tc.loadErr && args[0] == "load" {
					return errors.New("load failed")
				}
				return nil
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v", err)
			}
			if !reflect.DeepEqual((*calls)[len(*calls)-1], []string{"unload", "--all"}) {
				t.Fatalf("calls = %v", *calls)
			}
		})
	}
}

func TestLMStudioMissingCommand(t *testing.T) {
	old := LMSCommand
	t.Cleanup(func() { LMSCommand = old })
	LMSCommand = filepath.Join(t.TempDir(), "missing")
	_, err := LMStudio(context.Background(), "model", 4096, 128, func(string, ...string) error { t.Fatal("unexpected runner"); return nil })
	if err == nil || !strings.Contains(err.Error(), "lms") {
		t.Fatalf("error = %v", err)
	}
}

func TestLMStudioCommandFallback(t *testing.T) {
	old := LMSCommand
	t.Cleanup(func() { LMSCommand = old })
	LMSCommand = "lms"
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("PATH", t.TempDir())
	command := filepath.Join(home, ".lmstudio", "bin", "lms")
	if err := os.MkdirAll(filepath.Dir(command), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(command, []byte("#!/bin/sh\nexit 0\n"), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := lmsPath()
	if err != nil || got != command {
		t.Fatalf("path = %q, %v", got, err)
	}
}
