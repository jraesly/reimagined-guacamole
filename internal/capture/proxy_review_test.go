package capture

// Regression tests for the findings of the adversarial proxy review.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNullStreamOptionsDoesNotPanic(t *testing.T) {
	var seen []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseChunk("ok")+"data: [DONE]\n\n")
	}))
	defer up.Close()
	onTurn, wait := waitForTurn(t)
	p := newProxy(t, up, "/v1", Options{OnTurn: onTurn})
	srv := httptest.NewServer(p)
	defer srv.Close()

	body := chatBody("m", true, "null")
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	turn := wait()
	if turn.Status != "ok" || !turn.UsageInjected {
		t.Errorf("turn = %+v", turn)
	}
	var got map[string]any
	if err := json.Unmarshal(seen, &got); err != nil {
		t.Fatalf("upstream body not JSON: %v\n%s", err, seen)
	}
	so, _ := got["stream_options"].(map[string]any)
	if so["include_usage"] != true {
		t.Errorf("include_usage not injected over null: %s", seen)
	}
}

func TestInjectionPreservesEveryOtherByte(t *testing.T) {
	raw := []byte(`{"z_last":1,"model":"m","stream":true,  "messages":[{"role":"user","content":"hi"}],"temperature":0.2}` + "\n")
	out, ok := injectUsage(raw)
	if !ok {
		t.Fatal("injection failed")
	}
	want := `{"z_last":1,"model":"m","stream":true,  "messages":[{"role":"user","content":"hi"}],"temperature":0.2,"stream_options":{"include_usage":true}}` + "\n"
	if string(out) != want {
		t.Errorf("injected body changed unrelated bytes:\n got %s\nwant %s", out, want)
	}
	if _, err := ParseRequest(out); err != nil {
		t.Errorf("injected body no longer parses: %v", err)
	}
}

func TestRedirectIsRelayedNotFollowed(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/elsewhere" {
			t.Error("proxy followed the redirect")
		}
		http.Redirect(w, r, "/elsewhere", http.StatusFound)
	}))
	defer up.Close()
	p := newProxy(t, up, "", Options{})
	srv := httptest.NewServer(p)
	defer srv.Close()
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusFound || resp.Header.Get("Location") == "" {
		t.Errorf("client got %d %q, want the 302 itself", resp.StatusCode, resp.Header.Get("Location"))
	}
}

func TestTruncatedUpstreamIsNotDeliveredAsComplete(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseChunk("partial"))
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // drop the connection mid-stream
	}))
	defer up.Close()
	onTurn, wait := waitForTurn(t)
	p := newProxy(t, up, "/v1", Options{OnTurn: onTurn})
	srv := httptest.NewServer(p)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody("m", true, "")))
	if err != nil {
		t.Fatal(err)
	}
	_, readErr := io.ReadAll(resp.Body)
	resp.Body.Close()
	if readErr == nil {
		t.Error("client read the truncated stream as a clean end; it must see an error")
	}
	turn := wait()
	if turn.Status != "upstream-error" {
		t.Errorf("status = %q, want upstream-error", turn.Status)
	}
}

func TestEscapedPathAndConnectionNominatedHeaders(t *testing.T) {
	var gotPath, gotInternal, gotAE string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		gotInternal = r.Header.Get("X-Internal")
		gotAE = r.Header.Get("Accept-Encoding")
		_, _ = io.WriteString(w, "{}")
	}))
	defer up.Close()
	p := newProxy(t, up, "/v1/", Options{})
	srv := httptest.NewServer(p)
	defer srv.Close()
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/v1/models/a%2Fb", nil)
	req.Header.Set("Connection", "X-Internal")
	req.Header.Set("X-Internal", "secret")
	req.Header.Set("Accept-Encoding", "gzip")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if gotPath != "/v1/models/a%2Fb" {
		t.Errorf("upstream path = %q, want escaped form preserved without a doubled slash", gotPath)
	}
	if gotInternal != "" {
		t.Errorf("header nominated by Connection was forwarded: %q", gotInternal)
	}
	if gotAE != "" {
		t.Errorf("Accept-Encoding forwarded/added: %q", gotAE)
	}
}

func TestSessionHeaderSeparatesParallelAgents(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseChunk("ok")+"data: [DONE]\n\n")
	}))
	defer up.Close()
	var turns []Turn
	p := newProxy(t, up, "/v1", Options{OnTurn: func(t Turn) { turns = append(turns, t) }})
	srv := httptest.NewServer(p)
	defer srv.Close()
	for _, sess := range []string{"agent-a", "agent-b", "agent-a"} {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", bytes.NewReader(chatBody("m", true, "")))
		req.Header.Set("User-Agent", "same-ua")
		req.Header.Set(SessionHeader, sess)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
	}
	if !p.Wait(2 * time.Second) {
		t.Fatal("turns did not finish")
	}
	if len(turns) != 3 || !turns[0].FirstTurn || !turns[1].FirstTurn || turns[2].FirstTurn {
		t.Errorf("first-turn flags = %v %v %v (want true true false)", turns[0].FirstTurn, turns[1].FirstTurn, turns[2].FirstTurn)
	}
	if strings.HasPrefix(turns[0].Session, "same-ua") {
		t.Errorf("session key should come from %s, got %q", SessionHeader, turns[0].Session)
	}
}

func TestDumpFilesArePrivateAndNeverOverwritten(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"x"}}]}`)
	}))
	defer up.Close()
	dir := t.TempDir()
	onTurn, wait := waitForTurn(t)
	p := newProxy(t, up, "/v1", Options{OnTurn: onTurn, DumpDir: dir})
	srv := httptest.NewServer(p)
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody("m", false, "")))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	wait()
	entries, _ := os.ReadDir(dir)
	if len(entries) != 2 {
		t.Fatalf("dump files = %d, want request + response", len(entries))
	}
	for _, e := range entries {
		info, _ := e.Info()
		if info.Mode().Perm() != 0o600 {
			t.Errorf("%s mode = %o, want 0600", e.Name(), info.Mode().Perm())
		}
		if !strings.HasPrefix(e.Name(), p.runID+"-turn-1-") {
			t.Errorf("dump name %q lacks the run id prefix", e.Name())
		}
	}
	// A second proxy in the same directory gets its own run id and never
	// truncates an existing file.
	before, _ := os.ReadFile(filepath.Join(dir, entries[0].Name()))
	p2 := newProxy(t, up, "/v1", Options{DumpDir: dir, Now: func() time.Time { return time.Now().Add(time.Hour) }})
	srv2 := httptest.NewServer(p2)
	defer srv2.Close()
	resp, _ = http.Post(srv2.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody("m", false, "")))
	resp.Body.Close()
	p2.Wait(2 * time.Second)
	after, _ := os.ReadFile(filepath.Join(dir, entries[0].Name()))
	if !bytes.Equal(before, after) {
		t.Error("earlier dump file was modified")
	}
}
