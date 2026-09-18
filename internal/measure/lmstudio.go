package measure

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// LMStudioBase is the local API; tests can redirect it.
var LMStudioBase = "http://127.0.0.1:1234"

// LMSCommand is resolved on PATH, then in LM Studio's default installation.
var LMSCommand = "lms"

// LMStudioModel describes an installed model and its current load state.
type LMStudioModel struct {
	ID            string `json:"id"`
	Publisher     string `json:"publisher"`
	Arch          string `json:"arch"`
	Quant         string `json:"quantization"`
	Type          string `json:"type"`
	State         string `json:"state"`
	MaxContext    int    `json:"max_context_length"`
	LoadedContext int    `json:"loaded_context_length"`
}

// LMStudioAvailable reports whether the models API answers successfully.
func LMStudioAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, LMStudioBase+"/api/v0/models", nil)
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

// LMStudioModels lists installed models, including their context sizes.
func LMStudioModels(ctx context.Context) ([]LMStudioModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, LMStudioBase+"/api/v0/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("lmstudio: list models: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("lmstudio: list models: HTTP %d", resp.StatusCode)
	}
	var body struct {
		Data []LMStudioModel `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&body); err != nil {
		return nil, fmt.Errorf("lmstudio: list models: %w", err)
	}
	return body.Data, nil
}

// LMStudioID derives the API id from an installed repository name.
func LMStudioID(owner, repo string) string {
	return strings.TrimSuffix(strings.ToLower(repo), "-gguf")
}

// MatchLMStudio prefers an exact id, then a publisher-qualified partial id.
// Embedding models cannot perform the generation used for measurement.
func MatchLMStudio(models []LMStudioModel, owner, repo string) (LMStudioModel, bool) {
	id := LMStudioID(owner, repo)
	if id == "" {
		return LMStudioModel{}, false
	}
	for _, m := range models {
		if !strings.EqualFold(m.Type, "embeddings") && m.ID == id {
			return m, true
		}
	}
	for _, m := range models {
		if !strings.EqualFold(m.Type, "embeddings") && strings.EqualFold(m.Publisher, owner) && strings.Contains(strings.ToLower(m.ID), id) {
			return m, true
		}
	}
	// Catalog downloads (owner lmstudio-community) get ids of the form
	// "<vendor>/<name>", e.g. NVIDIA-Nemotron-3-Nano-4B-GGUF becomes
	// nvidia/nemotron-3-nano-4b; the slash-flattened id equals the repo key.
	for _, m := range models {
		if !strings.EqualFold(m.Type, "embeddings") && strings.ReplaceAll(strings.ToLower(m.ID), "/", "-") == id {
			return m, true
		}
	}
	return LMStudioModel{}, false
}

func lmsPath() (string, error) {
	path, err := exec.LookPath(LMSCommand)
	if err == nil {
		return path, nil
	}
	if LMSCommand == "lms" {
		if home, homeErr := os.UserHomeDir(); homeErr == nil {
			if path, err = exec.LookPath(filepath.Join(home, ".lmstudio", "bin", "lms")); err == nil {
				return path, nil
			}
		}
	}
	return "", fmt.Errorf("lmstudio: lms command %q not found; install the LM Studio CLI: %w", LMSCommand, err)
}

// LMStudio loads explicitly before generating so JIT defaults cannot silently
// select a huge context. A nil runner executes the CLI with the caller's context.
// Unloading is attempted after every load attempt, including failures.
func LMStudio(ctx context.Context, id string, numCtx, numPredict int, run func(name string, args ...string) error) (result Result, err error) {
	if id == "" || numCtx <= 0 || numPredict <= 0 {
		return result, errors.New("lmstudio: model, positive context and output token count required")
	}
	command, err := lmsPath()
	if err != nil {
		return result, err
	}
	injected := run != nil
	// lms prints progress bars; keep them out of probe's output and surface
	// the tail only when the command fails.
	quiet := func(c context.Context) func(name string, args ...string) error {
		return func(name string, args ...string) error {
			cmd := exec.CommandContext(c, name, args...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				tail := strings.TrimSpace(string(out))
				if len(tail) > 300 {
					tail = tail[len(tail)-300:]
				}
				return fmt.Errorf("%w: %s", err, tail)
			}
			return nil
		}
	}
	if run == nil {
		run = quiet(ctx)
	}
	defer func() {
		cleanup := run
		if !injected {
			// The generation deadline must not prevent unloading the model.
			cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
			defer cancel()
			cleanup = quiet(cleanupCtx)
		}
		if unloadErr := cleanup(command, "unload", "--all"); unloadErr != nil {
			err = errors.Join(err, fmt.Errorf("lmstudio: unload: %w", unloadErr))
		}
	}()
	if err := run(command, "load", id, "--context-length", strconv.Itoa(numCtx), "--gpu", "max", "-y"); err != nil {
		return result, fmt.Errorf("lmstudio: load %s: %w", id, err)
	}
	models, err := LMStudioModels(ctx)
	if err != nil {
		return result, err
	}
	loaded := false
	result.Model, result.Context = id, numCtx
	for _, m := range models {
		if m.ID != id {
			continue
		}
		if m.State != "loaded" {
			return result, fmt.Errorf("lmstudio: model %s is not loaded (state %q)", id, m.State)
		}
		loaded = true
		if m.LoadedContext != numCtx {
			result.Notes = append(result.Notes, fmt.Sprintf("loaded context %d differs from requested %d", m.LoadedContext, numCtx))
			// Calibration must describe the actual run, not the requested context.
			result.Context = m.LoadedContext
		}
		break
	}
	if !loaded {
		return result, fmt.Errorf("lmstudio: loaded model %s absent from models API", id)
	}
	body := map[string]any{"model": id, "messages": []map[string]string{{"role": "user", "content": Prompt}}, "max_tokens": numPredict, "stream": false, "temperature": 0}
	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, LMStudioBase+"/api/v0/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return result, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := Client.Do(req)
	if err != nil {
		return result, fmt.Errorf("lmstudio: completion: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return result, fmt.Errorf("lmstudio: completion: HTTP %d", resp.StatusCode)
	}
	var completion struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Stats *struct {
			TokensPerSecond *float64 `json:"tokens_per_second"`
			TTFT            *float64 `json:"time_to_first_token"`
			GenerationTime  *float64 `json:"generation_time"`
		} `json:"stats"`
		Runtime struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"runtime"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&completion); err != nil {
		return result, fmt.Errorf("lmstudio: completion: %w", err)
	}
	stats := completion.Stats
	if stats == nil || stats.TokensPerSecond == nil || stats.TTFT == nil || stats.GenerationTime == nil {
		return result, errors.New("lmstudio: response has no complete generation stats")
	}
	if *stats.TokensPerSecond <= 0 || completion.Usage.CompletionTokens <= 0 {
		return result, errors.New("lmstudio: response has no measured output tokens or positive stats.tokens_per_second")
	}
	result.OutputTokens = completion.Usage.CompletionTokens
	result.PromptTokens = completion.Usage.PromptTokens
	result.TokPerSec = *stats.TokensPerSecond
	result.TTFTSeconds = *stats.TTFT
	result.TotalSeconds = *stats.GenerationTime
	result.Runtime = strings.TrimSpace(completion.Runtime.Name + " " + completion.Runtime.Version)
	return result, nil
}
