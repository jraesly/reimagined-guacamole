package measure

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func fakeOllama(t *testing.T, thinkingSupported bool) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var calls []map[string]any
	mux := http.NewServeMux()
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(`{"version":"0.34.1"}`)) })
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		calls = append(calls, body)
		if _, hasThink := body["think"]; hasThink && !thinkingSupported {
			w.WriteHeader(400)
			_, _ = w.Write([]byte(`{"error":"\"m\" does not support thinking"}`))
			return
		}
		if body["model"] == "missing" {
			w.WriteHeader(404)
			_, _ = w.Write([]byte(`{"error":"model \"missing\" not found"}`))
			return
		}
		_, _ = w.Write([]byte(`{"eval_count":128,"eval_duration":11277000000,"prompt_eval_count":41,"prompt_eval_duration":937000000,"load_duration":11117000000,"total_duration":73303000000}`))
	})
	srv := httptest.NewServer(mux)
	OllamaBase = srv.URL
	return srv, &calls
}

func TestOllamaMeasuresAndDisablesThinking(t *testing.T) {
	srv, calls := fakeOllama(t, true)
	defer srv.Close()
	if !OllamaAvailable(context.Background()) {
		t.Fatal("fake server should be available")
	}
	r, err := Ollama(context.Background(), "qwen3.8:27b", 4096, 128)
	if err != nil {
		t.Fatal(err)
	}
	if r.OutputTokens != 128 || r.TokPerSec < 11.3 || r.TokPerSec > 11.4 || r.PromptTokPerSec < 43 || r.PromptTokPerSec > 44 {
		t.Errorf("result = %+v", r)
	}
	if r.LoadSeconds < 11 || r.LoadSeconds > 11.2 || !r.ThinkingDisabled {
		t.Errorf("result = %+v", r)
	}
	c := (*calls)[0]
	opts := c["options"].(map[string]any)
	if c["think"] != false || c["keep_alive"] != float64(0) || opts["num_ctx"] != float64(4096) || opts["num_predict"] != float64(128) || c["stream"] != false {
		t.Errorf("request = %v", c)
	}
}

func TestOllamaRetriesWithoutThinkForOldModels(t *testing.T) {
	srv, calls := fakeOllama(t, false)
	defer srv.Close()
	r, err := Ollama(context.Background(), "m", 4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(*calls) != 2 || r.ThinkingDisabled {
		t.Errorf("calls = %d, thinkingDisabled=%v", len(*calls), r.ThinkingDisabled)
	}
	if _, has := (*calls)[1]["think"]; has {
		t.Error("retry must omit the think field")
	}
}

func TestOllamaErrors(t *testing.T) {
	srv, _ := fakeOllama(t, true)
	defer srv.Close()
	if _, err := Ollama(context.Background(), "missing", 4096, 64); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("missing model: %v", err)
	}
	srv.Close()
	if OllamaAvailable(context.Background()) {
		t.Error("closed server should be unavailable")
	}
}
