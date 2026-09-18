package capture

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// newProxy builds a Proxy pointed at the given fake upstream server, using
// the upstream form (with or without a /v1 path) the caller asks for.
func newProxy(t *testing.T, upstream *httptest.Server, path string, opts Options) *Proxy {
	t.Helper()
	u, err := url.Parse(upstream.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	opts.Upstream = u
	p, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func chatBody(model string, stream bool, extra string) []byte {
	m := map[string]any{
		"model":    model,
		"stream":   stream,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	}
	raw, _ := json.Marshal(m)
	if extra == "" {
		return raw
	}
	var obj map[string]json.RawMessage
	_ = json.Unmarshal(raw, &obj)
	obj["stream_options"] = json.RawMessage(extra)
	out, _ := json.Marshal(obj)
	return out
}

func sseChunk(content string) string {
	return fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", content)
}

// waitForTurn returns an OnTurn callback and a matching wait function. The
// handler calls OnTurn after the response body has already reached the
// client (for a fixed-Content-Length response the client can finish reading
// before the handler goroutine gets to it), so a test that wants to inspect
// the recorded Turn must synchronize on it rather than read a plain
// variable right after its HTTP call returns.
func waitForTurn(t *testing.T) (onTurn func(Turn), wait func() Turn) {
	t.Helper()
	ch := make(chan Turn, 8)
	return func(tn Turn) { ch <- tn }, func() Turn {
		select {
		case tn := <-ch:
			return tn
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for OnTurn")
			return Turn{}
		}
	}
}

// -- byte-exact SSE pass-through -------------------------------------------

func TestProxySSEByteExact(t *testing.T) {
	want := sseChunk("Hel") + sseChunk("lo") + "data: [DONE]\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// write in odd chunk boundaries, unrelated to the SSE event boundaries
		for i := 0; i < len(want); i += 5 {
			end := i + 5
			if end > len(want) {
				end = len(want)
			}
			_, _ = w.Write([]byte(want[i:end]))
			w.(http.Flusher).Flush()
		}
	}))
	defer upstream.Close()

	p := newProxy(t, upstream, "/v1", Options{})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", true, ""))))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	if string(got) != want {
		t.Errorf("body mismatch:\n got=%q\nwant=%q", got, want)
	}
}

// -- TTFT and decode timing --------------------------------------------------

func TestProxyTTFTAndDecodeTiming(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		time.Sleep(150 * time.Millisecond)
		_, _ = w.Write([]byte(sseChunk("a")))
		fl.Flush()
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte(sseChunk("b")))
		fl.Flush()
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte(sseChunk("c")))
		fl.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		fl.Flush()
	}))
	defer upstream.Close()

	var got Turn
	var mu sync.Mutex
	p := newProxy(t, upstream, "/v1", Options{OnTurn: func(t Turn) { mu.Lock(); got = t; mu.Unlock() }})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", true, ""))))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if got.TTFTms.Value < 90 || got.TTFTms.Value > 210 {
		t.Errorf("TTFTms = %v, want ~150 (+-60)", got.TTFTms.Value)
	}
	if got.TotalMs.Value <= got.TTFTms.Value {
		t.Errorf("TotalMs %v should exceed TTFTms %v", got.TotalMs.Value, got.TTFTms.Value)
	}
	if got.PrefillShare.Value < 0.4 || got.PrefillShare.Value > 0.95 {
		t.Errorf("PrefillShare = %v, want in [0.4, 0.95]", got.PrefillShare.Value)
	}
}

// -- usage injection ----------------------------------------------------------

func TestProxyInjectsUsageWhenMissing(t *testing.T) {
	var sawStream map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var obj map[string]any
		_ = json.Unmarshal(body, &obj)
		sawStream, _ = obj["stream_options"].(map[string]any)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", true, ""))))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if sawStream == nil || sawStream["include_usage"] != true {
		t.Errorf("upstream did not see include_usage=true: %+v", sawStream)
	}
	if !got.UsageInjected {
		t.Error("Turn.UsageInjected should be true")
	}
}

func TestProxyDoesNotInjectWhenAlreadyPresent(t *testing.T) {
	reqBody := chatBody("m", true, `{"include_usage":true}`)
	var sawBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if string(sawBody) != string(reqBody) {
		t.Errorf("body was altered:\n got=%s\nwant=%s", sawBody, reqBody)
	}
	if got.UsageInjected {
		t.Error("Turn.UsageInjected should be false")
	}
}

// -- non-streaming request ------------------------------------------------

func TestProxyNonStreaming(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi there"}}],"usage":{"prompt_tokens":42,"completion_tokens":3}}`))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", false, ""))))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if !strings.Contains(string(body), "hi there") {
		t.Errorf("client did not receive content: %s", body)
	}
	if got.Status != "ok" {
		t.Errorf("Status = %q, want ok", got.Status)
	}
	if got.PromptTokens.Source != model.Measured || got.PromptTokens.Value != 42 {
		t.Errorf("PromptTokens = %+v, want measured 42", got.PromptTokens)
	}
	if got.OutputTokens == nil || got.OutputTokens.Value != 3 {
		t.Errorf("OutputTokens = %+v", got.OutputTokens)
	}
}

// -- upstream 500 -----------------------------------------------------------

func TestProxyUpstreamError(t *testing.T) {
	errBody := `{"error":{"message":"boom"}}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(errBody))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", false, ""))))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("client status = %d, want 500", resp.StatusCode)
	}
	if string(body) != errBody {
		t.Errorf("body mismatch: got=%q want=%q", body, errBody)
	}
	if got.Status != "upstream-error" {
		t.Errorf("Status = %q, want upstream-error", got.Status)
	}
	if got.HTTPStatus != 500 {
		t.Errorf("HTTPStatus = %d, want 500", got.HTTPStatus)
	}
}

// -- client cancel ------------------------------------------------------------

func TestProxyClientCancelAborts(t *testing.T) {
	upstreamCtxCancelled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseChunk("a")))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(upstreamCtxCancelled)
		case <-time.After(5 * time.Second):
		}
	}))
	defer upstream.Close()

	var got Turn
	var mu sync.Mutex
	p := newProxy(t, upstream, "/v1", Options{OnTurn: func(t Turn) { mu.Lock(); got = t; mu.Unlock() }})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, proxySrv.URL+"/v1/chat/completions", strings.NewReader(string(chatBody("m", true, ""))))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 64)
	_, _ = resp.Body.Read(buf) // read the first chunk so the request is underway
	cancel()
	resp.Body.Close()

	select {
	case <-upstreamCtxCancelled:
	case <-time.After(1 * time.Second):
		t.Fatal("upstream did not observe context cancellation within 1s")
	}

	// give the proxy handler a moment to finish and record the turn
	deadline := time.After(2 * time.Second)
	for {
		mu.Lock()
		status := got.Status
		mu.Unlock()
		if status == "aborted" {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("Status = %q, want aborted", status)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// -- non-JSON and oversize bodies --------------------------------------------

func TestProxyNonJSONBodyForwardedUnaccounted(t *testing.T) {
	const raw = "not json at all"
	var sawBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if string(sawBody) != raw {
		t.Errorf("upstream saw %q, want %q", sawBody, raw)
	}
	if got.Status != "unaccounted" {
		t.Errorf("Status = %q, want unaccounted", got.Status)
	}
}

func TestProxyOversizeBodyForwardedUnaccounted(t *testing.T) {
	big := strings.Repeat("x", 200)
	body := chatBody("m", false, "")
	// inflate the body past MaxBody by padding a message field
	padded := strings.Replace(string(body), `"hi"`, `"`+big+`"`, 1)

	var sawBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		sawBody = string(b)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{MaxBody: 50, OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(padded))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	if sawBody != padded {
		t.Errorf("upstream body mismatch:\n got=%q\nwant=%q", sawBody, padded)
	}
	if got.Status != "unaccounted" {
		t.Errorf("Status = %q, want unaccounted", got.Status)
	}
}

// -- pass-through -------------------------------------------------------------

func TestProxyGetModelsPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer upstream.Close()

	var turnCalled bool
	p := newProxy(t, upstream, "/v1", Options{OnTurn: func(t Turn) { turnCalled = true }})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Get(proxySrv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if string(body) != `{"data":[]}` {
		t.Errorf("body = %q", body)
	}
	if turnCalled {
		t.Error("OnTurn should not fire for a pass-through request")
	}
}

// -- path rule ----------------------------------------------------------------

func TestProxyPathRuleWithV1Upstream(t *testing.T) {
	var sawPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer upstream.Close()

	p := newProxy(t, upstream, "/v1", Options{})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", false, ""))))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	if sawPath != "/v1/chat/completions" {
		t.Errorf("upstream saw path %q, want /v1/chat/completions", sawPath)
	}
}

func TestProxyPathRuleWithBareUpstream(t *testing.T) {
	var sawPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer upstream.Close()

	p := newProxy(t, upstream, "", Options{})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(chatBody("m", false, ""))))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	if sawPath != "/v1/chat/completions" {
		t.Errorf("upstream saw path %q, want /v1/chat/completions", sawPath)
	}
}

// -- prefix tracking ----------------------------------------------------------

func TestProxyPrefixTrackingAcrossTurns(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":100,"completion_tokens":1}}`))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	post := func(ua string, body []byte) Turn {
		req, _ := http.NewRequest(http.MethodPost, proxySrv.URL+"/v1/chat/completions", strings.NewReader(string(body)))
		req.Header.Set("User-Agent", ua)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
		return nextTurn()
	}

	sysA := `{"role":"system","content":"you are helpful"}`
	sysB := `{"role":"system","content":"you are a pirate"}`
	user1 := `{"role":"user","content":"hi"}`
	user2 := `{"role":"user","content":"now what"}`

	msgBody := func(sys, user string, more ...string) []byte {
		msgs := []json.RawMessage{json.RawMessage(sys), json.RawMessage(user)}
		for _, m := range more {
			msgs = append(msgs, json.RawMessage(m))
		}
		b, _ := json.Marshal(map[string]any{"model": "m", "stream": false, "messages": msgs})
		return b
	}

	var turns []Turn
	turns = append(turns, post("harness-A", msgBody(sysA, user1)))
	turns = append(turns, post("harness-A", msgBody(sysA, user1, `{"role":"assistant","content":"hello"}`, user2)))
	turns = append(turns, post("harness-A", msgBody(sysB, user1)))
	turns = append(turns, post("harness-B", msgBody(sysA, user1)))

	if len(turns) != 4 {
		t.Fatalf("got %d turns, want 4", len(turns))
	}
	if !turns[0].FirstTurn {
		t.Errorf("turn 0: FirstTurn = false, want true")
	}
	if turns[1].FirstTurn || turns[1].PrefixBreak != nil {
		t.Errorf("turn 1 (append-only): FirstTurn=%v PrefixBreak=%+v", turns[1].FirstTurn, turns[1].PrefixBreak)
	}
	prevCanon := Canonicalize(mustParse(t, msgBody(sysA, user1))).Canonical
	if turns[1].PrefixBytes != len(prevCanon) {
		t.Errorf("turn 1 PrefixBytes = %d, want %d", turns[1].PrefixBytes, len(prevCanon))
	}
	if turns[2].PrefixBreak == nil || turns[2].PrefixBreak.Segment != "system[0].content" {
		t.Errorf("turn 2 (changed system prompt): PrefixBreak = %+v", turns[2].PrefixBreak)
	}
	if !turns[3].FirstTurn {
		t.Errorf("turn 3 (different User-Agent): FirstTurn = false, want true")
	}
}

func mustParse(t *testing.T, body []byte) *Request {
	t.Helper()
	r, err := ParseRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// -- apportionment --------------------------------------------------------

func TestProxyApportionment(t *testing.T) {
	sys := `{"role":"system","content":"` + strings.Repeat("s", 100) + `"}`
	user := `{"role":"user","content":"` + strings.Repeat("u", 300) + `"}`
	body, _ := json.Marshal(map[string]any{
		"model": "m", "stream": false,
		"messages": []json.RawMessage{json.RawMessage(sys), json.RawMessage(user)},
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1000,"completion_tokens":1}}`))
	}))
	defer upstream.Close()

	onTurn, nextTurn := waitForTurn(t)
	p := newProxy(t, upstream, "/v1", Options{OnTurn: onTurn})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	got := nextTurn()

	sum := got.SystemTokens.Value + got.ToolsTokens.Value + got.HistoryTokens.Value + got.CurrentTokens.Value
	if diff := sum - 1000; diff > 1 || diff < -1 {
		t.Errorf("sum of parts = %v, want ~1000", sum)
	}
	for name, v := range map[string]Value{"system": got.SystemTokens, "tools": got.ToolsTokens, "history": got.HistoryTokens, "current": got.CurrentTokens} {
		if v.Source != model.Inferred {
			t.Errorf("%s source = %q, want inferred", name, v.Source)
		}
	}
	if got.PromptTokens.Source != model.Measured {
		t.Errorf("PromptTokens source = %q, want measured", got.PromptTokens.Source)
	}
}

// -- DumpDir ------------------------------------------------------------------

func TestProxyDumpDirRedactsSecrets(t *testing.T) {
	secret := "sk-abcdefghijklmnopqrstuvwxyz"
	body, _ := json.Marshal(map[string]any{
		"model": "m", "stream": false,
		"messages": []map[string]any{{"role": "user", "content": "my key is " + secret}},
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"the key is ` + secret + `"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer upstream.Close()

	dir := t.TempDir()
	done := make(chan struct{})
	p := newProxy(t, upstream, "/v1", Options{DumpDir: dir, OnTurn: func(t Turn) { close(done) }})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	resp, err := http.Post(proxySrv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	<-done

	reqPath := filepath.Join(dir, p.runID+"-turn-1-request.json")
	respPath := filepath.Join(dir, p.runID+"-turn-1-response.txt")
	reqData, err := os.ReadFile(reqPath)
	if err != nil {
		t.Fatalf("request dump missing: %v", err)
	}
	respData, err := os.ReadFile(respPath)
	if err != nil {
		t.Fatalf("response dump missing: %v", err)
	}
	if strings.Contains(string(reqData), secret) {
		t.Errorf("request dump leaked the secret: %s", reqData)
	}
	if strings.Contains(string(respData), secret) {
		t.Errorf("response dump leaked the secret: %s", respData)
	}
}

// -- Accept-Encoding ------------------------------------------------------

func TestProxyStripsAcceptEncoding(t *testing.T) {
	// The client asks for something Go's transport would never negotiate on
	// its own, so seeing it upstream proves the client's header leaked
	// through instead of being deleted per rule 4. (Go's transport may add
	// its own "Accept-Encoding: gzip" once we delete the client's; that is
	// the transparent decompression the rule calls for, not a leak.)
	const clientValue = "identity;q=0.1, br;q=0.9"
	var sawAE string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAE = r.Header.Get("Accept-Encoding")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer upstream.Close()

	p := newProxy(t, upstream, "/v1", Options{})
	proxySrv := httptest.NewServer(p)
	defer proxySrv.Close()

	req, _ := http.NewRequest(http.MethodPost, proxySrv.URL+"/v1/chat/completions", strings.NewReader(string(chatBody("m", false, ""))))
	req.Header.Set("Accept-Encoding", clientValue)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	if sawAE == clientValue {
		t.Errorf("upstream saw the client's Accept-Encoding %q forwarded verbatim", sawAE)
	}
}
