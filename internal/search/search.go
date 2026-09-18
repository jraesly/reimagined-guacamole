// Package search finds candidate models for a task online: Hugging Face's
// model API (JSON) and ollama.com's search page (HTML, parsed tolerantly).
// It returns references that fit can resolve; it never downloads weights.
package search

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// Endpoints are variables so tests can point them at httptest servers.
var (
	HFBase     = "https://huggingface.co"
	OllamaSite = "https://ollama.com"
	Client     = &http.Client{Timeout: 30 * time.Second}
	UserAgent  = "probe/0.1 (+https://github.com/jraesly/reimagined-guacamole)"
)

// Candidate is one search hit.
type Candidate struct {
	Ref          string   `json:"ref"` // ollama:name:latest or hf:owner/repo
	Name         string   `json:"name"`
	Owner        string   `json:"owner"`
	Source       string   `json:"source"` // "huggingface" or "ollama.com"
	Description  string   `json:"description,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
	Sizes        []string `json:"sizes,omitempty"`
	Downloads    int64    `json:"downloads"`
	Likes        int64    `json:"likes,omitempty"`
	Cloud        bool     `json:"cloud"` // ollama.com cloud-only entry: no local weights
	Baseline     bool     `json:"baseline"`
	FitsOffline  bool     `json:"fits_offline"` // false for media tasks: listing only, no header fit
}

// taskQuery maps a probe task onto each source's vocabulary.
type taskQuery struct {
	hfPipeline string
	hfGGUF     bool // restrict to GGUF repos (LLM tasks)
	ollamaQ    string
	ollamaCap  string
	fits       bool // LLM tasks can be fitted from their headers
}

var taskQueries = map[string]taskQuery{
	"coding":    {"text-generation", true, "coding", "tools", true},
	"agent":     {"text-generation", true, "agent", "tools", true},
	"chat":      {"text-generation", true, "", "", true},
	"vision":    {"image-text-to-text", true, "", "vision", true},
	"embedding": {"feature-extraction", true, "", "embedding", true},
	"image":     {"text-to-image", false, "", "", false},
	"video":     {"text-to-video", false, "", "", false},
	"tts":       {"text-to-speech", false, "", "", false},
	"speech":    {"automatic-speech-recognition", false, "", "", false},
}

// Tasks lists the searchable task names.
func Tasks() []string {
	out := make([]string, 0, len(taskQueries))
	for t := range taskQueries {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// Find queries both sources for task and merges the results, baseline
// publishers first, then by downloads. variants keeps community fine-tunes.
func Find(ctx context.Context, task string, limit int, variants bool) ([]Candidate, []error) {
	q, ok := taskQueries[strings.ToLower(task)]
	if !ok {
		return nil, []error{fmt.Errorf("unknown task %q; searchable tasks: %s", task, strings.Join(Tasks(), ", "))}
	}
	var out []Candidate
	var errs []error
	hf, err := HuggingFace(ctx, q, limit*4)
	if err != nil {
		errs = append(errs, err)
	}
	out = append(out, hf...)
	if q.fits { // ollama.com only serves LLM/VLM/embedding weights
		ol, err := Ollama(ctx, q, limit*2)
		if err != nil {
			errs = append(errs, err)
		}
		out = append(out, ol...)
	}
	filtered := out[:0]
	for _, c := range out {
		if c.Cloud || (!variants && !c.Baseline) {
			continue
		}
		filtered = append(filtered, c)
	}
	sort.SliceStable(filtered, func(i, j int) bool {
		if filtered[i].Baseline != filtered[j].Baseline {
			return filtered[i].Baseline
		}
		return filtered[i].Downloads > filtered[j].Downloads
	})
	if len(filtered) > limit {
		filtered = filtered[:limit]
	}
	return filtered, errs
}

type hfModel struct {
	ID        string   `json:"id"`
	Downloads int64    `json:"downloads"`
	Likes     int64    `json:"likes"`
	Tags      []string `json:"tags"`
}

// HuggingFace queries the model API for the task's pipeline tag.
func HuggingFace(ctx context.Context, q taskQuery, limit int) ([]Candidate, error) {
	v := url.Values{}
	v.Set("pipeline_tag", q.hfPipeline)
	v.Set("sort", "trendingScore")
	v.Set("direction", "-1")
	v.Set("limit", strconv.Itoa(limit))
	if q.hfGGUF {
		v.Set("filter", "gguf")
	}
	body, err := get(ctx, HFBase+"/api/models?"+v.Encode())
	if err != nil {
		return nil, fmt.Errorf("huggingface search: %w", err)
	}
	var models []hfModel
	if err := json.Unmarshal(body, &models); err != nil {
		return nil, fmt.Errorf("huggingface search: %w", err)
	}
	var out []Candidate
	for _, m := range models {
		owner, repo, ok := strings.Cut(m.ID, "/")
		if !ok {
			continue
		}
		baseline, _ := model.Classify(owner, repo)
		c := Candidate{Ref: "hf:" + m.ID, Name: m.ID, Owner: owner, Source: "huggingface",
			Downloads: m.Downloads, Likes: m.Likes, Baseline: baseline, FitsOffline: q.fits}
		for _, t := range m.Tags {
			switch t {
			case "gguf", "safetensors", "mlx", "vision", "tools", "thinking", "conversational":
				c.Capabilities = append(c.Capabilities, t)
			}
		}
		out = append(out, c)
	}
	return out, nil
}

// Ollama queries ollama.com's search page and parses the result list.
func Ollama(ctx context.Context, q taskQuery, limit int) ([]Candidate, error) {
	v := url.Values{}
	if q.ollamaQ != "" {
		v.Set("q", q.ollamaQ)
	}
	if q.ollamaCap != "" {
		v.Set("c", q.ollamaCap)
	}
	body, err := get(ctx, OllamaSite+"/search?"+v.Encode())
	if err != nil {
		return nil, fmt.Errorf("ollama.com search: %w", err)
	}
	out := ParseOllamaSearch(string(body))
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

var (
	ollamaEntryRe = regexp.MustCompile(`(?s)<a href="/library/([a-z0-9._-]+)"[^>]*>(.*?)</a>`)
	ollamaDescRe  = regexp.MustCompile(`(?s)<p[^>]*>(.*?)</p>`)
	ollamaSpanRe  = regexp.MustCompile(`(?s)<span[^>]*>\s*([^<]*?)\s*</span>`)
	// "<span>141.3K</span> <span class="hidden sm:flex">&nbsp;Pulls</span>" as served in 2026.
	ollamaPullsRe = regexp.MustCompile(`([\d.]+[KMB]?)\s*(?:</span>\s*)?(?:<span[^>]*>)?(?:&nbsp;|\s)*Pulls`)
	ollamaSizeRe  = regexp.MustCompile(`^(?:\d+x)?\d+(?:\.\d+)?[bm]$|^e\d+b$`)
	tagRe         = regexp.MustCompile(`<[^>]+>`)
)

var ollamaCapabilities = map[string]bool{"vision": true, "tools": true, "thinking": true, "cloud": true, "embedding": true, "audio": true}

// ParseOllamaSearch extracts entries from ollama.com's library/search HTML.
// It keys on the `/library/<name>` links and reads what it can from each
// block; anything it cannot find is left empty rather than guessed.
func ParseOllamaSearch(page string) []Candidate {
	var out []Candidate
	seen := map[string]bool{}
	for _, m := range ollamaEntryRe.FindAllStringSubmatch(page, -1) {
		name, block := m[1], m[2]
		if seen[name] {
			continue
		}
		seen[name] = true
		c := Candidate{Ref: "ollama:" + name + ":latest", Name: name, Owner: "library", Source: "ollama.com", Baseline: true, FitsOffline: true}
		if d := ollamaDescRe.FindStringSubmatch(block); d != nil {
			c.Description = strings.TrimSpace(html.UnescapeString(tagRe.ReplaceAllString(d[1], "")))
		}
		pulls := ""
		if p := ollamaPullsRe.FindStringSubmatch(block); p != nil {
			pulls = strings.ToLower(p[1])
			c.Downloads = parseCount(p[1])
		}
		for _, s := range ollamaSpanRe.FindAllStringSubmatch(block, -1) {
			text := strings.ToLower(strings.TrimSpace(s[1]))
			switch {
			case text == pulls && pulls != "":
				// the pull count ("2M") would otherwise read as a size
			case ollamaCapabilities[text]:
				c.Capabilities = append(c.Capabilities, text)
				if text == "cloud" {
					c.Cloud = true
				}
			case ollamaSizeRe.MatchString(text):
				c.Sizes = append(c.Sizes, text)
			}
		}
		// A cloud-only entry has no local sizes; a hybrid entry (cloud +
		// sizes) can still be pulled.
		if c.Cloud && len(c.Sizes) > 0 {
			c.Cloud = false
		}
		out = append(out, c)
	}
	return out
}

func parseCount(s string) int64 {
	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "K"):
		mult, s = 1_000, strings.TrimSuffix(s, "K")
	case strings.HasSuffix(s, "M"):
		mult, s = 1_000_000, strings.TrimSuffix(s, "M")
	case strings.HasSuffix(s, "B"):
		mult, s = 1_000_000_000, strings.TrimSuffix(s, "B")
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return int64(f * float64(mult))
}

func get(ctx context.Context, u string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", UserAgent)
	resp, err := Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d from %s", resp.StatusCode, u)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 8<<20))
}
