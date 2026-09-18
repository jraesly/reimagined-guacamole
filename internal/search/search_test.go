package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ollamaPage is a trimmed copy of ollama.com/search markup as served on
// 2026-09-17: one hybrid entry, one cloud-only entry, one plain entry.
const ollamaPage = `<ul>
<li><a href="/library/qwen3.8" class="group w-full"> <div title="qwen3.8"> <h2><span >qwen3.8</span></h2>
<p class="max-w-lg">Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon &amp; agentic tasks.</p></div>
<div><span class="cap">vision</span> <span class="cap">tools</span> <span class="cap">thinking</span> <span class="cap">cloud</span> <span class="size">27b</span></div>
<p class="meta"><span >2M</span> <span class="hidden sm:flex">&nbsp;Pulls</span> <span>12</span> Tags Updated 1 month ago</p></a></li>
<li><a href="/library/glm-5.3" class="group w-full"><span >glm-5.3</span><p>Z.ai's flagship model.</p>
<span>tools</span><span>thinking</span><span>cloud</span><span>48.4K</span> Pulls</a></li>
<li><a href="/library/laguna-xs-2.1" class="group w-full"><span >laguna-xs-2.1</span><p>Built for agentic coding on a local machine.</p>
<span>tools</span><span>thinking</span><span>111.2K</span> Pulls</a></li>
<li><a href="/library/qwen3.8" class="dup"><span>qwen3.8</span></a></li>
</ul>`

func TestParseOllamaSearch(t *testing.T) {
	got := ParseOllamaSearch(ollamaPage)
	if len(got) != 3 {
		t.Fatalf("entries = %d: %+v", len(got), got)
	}
	q := got[0]
	if q.Name != "qwen3.8" || q.Ref != "ollama:qwen3.8:latest" || q.Cloud || q.Downloads != 2_000_000 {
		t.Errorf("qwen3.8 = %+v", q)
	}
	if !strings.Contains(q.Description, "long-horizon & agentic") {
		t.Errorf("description not unescaped: %q", q.Description)
	}
	if strings.Join(q.Capabilities, ",") != "vision,tools,thinking,cloud" || strings.Join(q.Sizes, ",") != "27b" {
		t.Errorf("caps=%v sizes=%v", q.Capabilities, q.Sizes)
	}
	if g := got[1]; !g.Cloud || g.Downloads != 48_400 {
		t.Errorf("cloud-only glm = %+v", g)
	}
	if l := got[2]; l.Cloud || l.Downloads != 111_200 || len(l.Sizes) != 0 {
		t.Errorf("laguna = %+v", l)
	}
	if len(ParseOllamaSearch("<html>nothing here</html>")) != 0 {
		t.Error("no entries expected")
	}
}

func TestParseCount(t *testing.T) {
	for in, want := range map[string]int64{"2M": 2_000_000, "48.4K": 48_400, "1.5B": 1_500_000_000, "731": 731, "x": 0} {
		if got := parseCount(in); got != want {
			t.Errorf("parseCount(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestFindMergesFiltersAndSorts(t *testing.T) {
	var hfQuery string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		hfQuery = r.URL.RawQuery
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"id": "OBLITERATUS/Qwen3.8-27B-OBLITERATED", "downloads": 9_000_000, "likes": 1, "tags": []string{"gguf", "uncensored"}},
			{"id": "unsloth/Qwen3.8-27B-GGUF", "downloads": 500_000, "likes": 2, "tags": []string{"gguf", "vision"}},
			{"id": "bartowski/Foo-GGUF", "downloads": 100, "likes": 0, "tags": []string{"gguf"}},
			{"id": "noslash", "downloads": 1},
		})
	})
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(ollamaPage)) })
	srv := httptest.NewServer(mux)
	defer srv.Close()
	HFBase, OllamaSite = srv.URL, srv.URL

	got, errs := Find(context.Background(), "coding", 10, false)
	if len(errs) != 0 {
		t.Fatal(errs)
	}
	if !strings.Contains(hfQuery, "filter=gguf") || !strings.Contains(hfQuery, "pipeline_tag=text-generation") {
		t.Errorf("hf query = %s", hfQuery)
	}
	names := make([]string, 0, len(got))
	for _, c := range got {
		names = append(names, c.Name)
	}
	// Baselines only, cloud-only dropped, sorted by downloads within baselines.
	want := "qwen3.8,unsloth/Qwen3.8-27B-GGUF,laguna-xs-2.1,bartowski/Foo-GGUF"
	if strings.Join(names, ",") != want {
		t.Errorf("order = %s\nwant  %s", strings.Join(names, ","), want)
	}
	// With variants the OBLITERATED repo appears, after the baselines.
	got, _ = Find(context.Background(), "coding", 10, true)
	if got[len(got)-1].Name != "OBLITERATUS/Qwen3.8-27B-OBLITERATED" || got[len(got)-1].Baseline {
		t.Errorf("variant placement: %+v", got[len(got)-1])
	}
	// Limit applies; media tasks skip ollama.com and are listing-only.
	if got, _ := Find(context.Background(), "coding", 2, false); len(got) != 2 {
		t.Errorf("limit not applied: %d", len(got))
	}
	got, _ = Find(context.Background(), "image", 10, true)
	for _, c := range got {
		if c.Source == "ollama.com" || c.FitsOffline {
			t.Errorf("image search should be HF-only and listing-only: %+v", c)
		}
	}
	if !strings.Contains(hfQuery, "pipeline_tag=text-to-image") || strings.Contains(hfQuery, "filter=gguf") {
		t.Errorf("image query = %s", hfQuery)
	}
	if _, errs := Find(context.Background(), "juggling", 5, false); len(errs) != 1 {
		t.Error("unknown task should error")
	}
}

func TestFindReportsSourceErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(500) }))
	defer srv.Close()
	HFBase, OllamaSite = srv.URL, srv.URL
	got, errs := Find(context.Background(), "chat", 5, false)
	if len(got) != 0 || len(errs) != 2 {
		t.Errorf("got=%v errs=%v", got, errs)
	}
}
