// Package measure runs short generations against a local model server and
// reports what the server itself measured.
package measure

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaBase is the local Ollama API; a variable so tests can redirect it.
var OllamaBase = "http://127.0.0.1:11434"

// Client has no overall timeout because loading a 20 GB model can take a
// minute; the caller's context bounds the call.
var Client = &http.Client{}

// Prompt is a small coding task so generation resembles agent use.
const Prompt = "Write a Python function that parses a semver string into a (major, minor, patch) tuple with input validation. Code only, no explanation."

// Result is what Ollama reported for one run. Rates are measured by the
// server, not by probe's clock.
type Result struct {
	Model            string
	Context          int
	OutputTokens     int
	TokPerSec        float64
	PromptTokens     int
	PromptTokPerSec  float64
	LoadSeconds      float64
	TotalSeconds     float64
	ThinkingDisabled bool
	TTFTSeconds      float64
	Runtime          string
	Notes            []string
}

// OllamaAvailable reports whether the local API answers.
func OllamaAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, OllamaBase+"/api/version", nil)
	if err != nil {
		return false
	}
	resp, err := Client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

type generateResp struct {
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	LoadDuration       int64  `json:"load_duration"`
	TotalDuration      int64  `json:"total_duration"`
	Error              string `json:"error"`
}

// Ollama generates numPredict tokens with the given context size and returns
// the server's timings. Thinking is disabled when the model supports it so
// the measurement is decode speed, not reasoning length.
func Ollama(ctx context.Context, model string, numCtx, numPredict int) (Result, error) {
	res, err := ollamaOnce(ctx, model, numCtx, numPredict, true)
	if err != nil && strings.Contains(err.Error(), "think") {
		res, err = ollamaOnce(ctx, model, numCtx, numPredict, false)
	}
	return res, err
}

func ollamaOnce(ctx context.Context, model string, numCtx, numPredict int, disableThink bool) (Result, error) {
	body := map[string]any{
		"model": model, "prompt": Prompt, "stream": false, "keep_alive": 0,
		"options": map[string]any{"num_ctx": numCtx, "num_predict": numPredict, "temperature": 0},
	}
	if disableThink {
		body["think"] = false
	}
	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, OllamaBase+"/api/generate", bytes.NewReader(raw))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := Client.Do(req)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return Result{}, err
	}
	var g generateResp
	if err := json.Unmarshal(data, &g); err != nil {
		return Result{}, fmt.Errorf("ollama: %s: %w", strings.TrimSpace(string(data)), err)
	}
	if g.Error != "" {
		return Result{}, errors.New("ollama: " + g.Error)
	}
	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("ollama: HTTP %d", resp.StatusCode)
	}
	if g.EvalCount == 0 || g.EvalDuration == 0 {
		return Result{}, errors.New("ollama: response has no eval_count/eval_duration")
	}
	r := Result{
		Model: model, Context: numCtx, OutputTokens: g.EvalCount,
		TokPerSec:        float64(g.EvalCount) / (float64(g.EvalDuration) / 1e9),
		PromptTokens:     g.PromptEvalCount,
		LoadSeconds:      float64(g.LoadDuration) / 1e9,
		TotalSeconds:     float64(g.TotalDuration) / 1e9,
		ThinkingDisabled: disableThink,
	}
	if g.PromptEvalDuration > 0 {
		r.PromptTokPerSec = float64(g.PromptEvalCount) / (float64(g.PromptEvalDuration) / 1e9)
	}
	return r, nil
}

// Timeout is a sensible bound for one measurement run including model load.
const Timeout = 10 * time.Minute
