package capture

import (
	"encoding/json"
	"strings"
	"testing"
)

func req(t *testing.T, tools []string, msgs ...string) *Request {
	t.Helper()
	body := map[string]any{"model": "m", "stream": true}
	var ms []json.RawMessage
	for _, m := range msgs {
		ms = append(ms, json.RawMessage(m))
	}
	body["messages"] = ms
	if tools != nil {
		var ts []json.RawMessage
		for _, tl := range tools {
			ts = append(ts, json.RawMessage(tl))
		}
		body["tools"] = ts
	}
	raw, _ := json.Marshal(body)
	r, err := ParseRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

const (
	sys      = `{"role":"system","content":"You are OpenCode. Current time: 22:41:03"}`
	sysLater = `{"role":"system","content":"You are OpenCode. Current time: 22:41:19"}`
	user1    = `{"role":"user","content":"fix the bug"}`
	asst1    = `{"role":"assistant","content":null,"tool_calls":[{"id":"1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}`
	toolRes  = `{"role":"tool","tool_call_id":"1","content":"file contents"}`
	user2    = `{"role":"user","content":"now add tests"}`
	toolA    = `{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object"}}}`
	toolB    = `{"type":"function","function":{"name":"write_file","description":"Write","parameters":{"type":"object"}}}`
)

func TestCountCharsSplitsPromptParts(t *testing.T) {
	r := req(t, []string{toolA, toolB}, sys, user1, asst1, toolRes, user2)
	p := Canonicalize(r)
	c, tools := CountChars(p)
	if c.System == 0 || c.Tools == 0 || c.History == 0 || c.Current != len("user:now add tests") {
		t.Errorf("chars = %+v", c)
	}
	if c.Total != c.System+c.Tools+c.History+c.Current {
		t.Errorf("total mismatch: %+v", c)
	}
	if len(tools) != 2 || tools[0].Chars < tools[1].Chars || tools[0].Name != "read_file" {
		t.Errorf("tools = %+v (want sorted desc, read_file largest)", tools)
	}
	// history must include the assistant tool_calls segment
	if len(p.Segments) != 2+1+1+2+1+1 {
		t.Errorf("segments = %d: %+v", len(p.Segments), p.Segments)
	}
}

func TestApportionAndEstimate(t *testing.T) {
	if got := Apportion(1000, 250, 1000); got != 250 {
		t.Errorf("Apportion = %v", got)
	}
	if got := Apportion(1000, 250, 0); got != 0 {
		t.Errorf("Apportion with zero total = %v", got)
	}
	if got := EstimateTokens(400); got != 100 {
		t.Errorf("EstimateTokens = %v", got)
	}
}

func TestCommonPrefixAppendOnlyHasNoBreak(t *testing.T) {
	prev := Canonicalize(req(t, []string{toolA}, sys, user1))
	cur := Canonicalize(req(t, []string{toolA}, sys, user1, asst1, toolRes, user2))
	n, br := CommonPrefix(prev, cur)
	if br != nil || n != len(prev.Canonical) {
		t.Errorf("append-only: n=%d br=%+v", n, br)
	}
	if n, br := CommonPrefix(prev, prev); br != nil || n != len(prev.Canonical) {
		t.Errorf("identical: n=%d br=%+v", n, br)
	}
}

func TestCommonPrefixTimestampInSystemPrompt(t *testing.T) {
	prev := Canonicalize(req(t, []string{toolA}, sys, user1))
	cur := Canonicalize(req(t, []string{toolA}, sysLater, user1, asst1))
	n, br := CommonPrefix(prev, cur)
	if br == nil {
		t.Fatal("expected a break")
	}
	if br.Segment != "system[0].content" || br.Reason != "changed" {
		t.Errorf("break = %+v", br)
	}
	if !strings.Contains(br.Excerpt, "Current time: 22:41:") || !strings.Contains(br.Excerpt, "|") {
		t.Errorf("excerpt = %q", br.Excerpt)
	}
	// the shared prefix is the tool plus everything up to the changed digit
	if n <= len(compactJSON(json.RawMessage(toolA))) {
		t.Errorf("prefix too short: %d", n)
	}
}

func TestCommonPrefixReorderedTools(t *testing.T) {
	prev := Canonicalize(req(t, []string{toolA, toolB}, sys, user1))
	cur := Canonicalize(req(t, []string{toolB, toolA}, sys, user1))
	n, br := CommonPrefix(prev, cur)
	if br == nil || br.Reason != "reordered" || !strings.HasPrefix(br.Segment, "tools[0]") {
		t.Errorf("n=%d break = %+v", n, br)
	}
}

func TestCommonPrefixRemovedTail(t *testing.T) {
	prev := Canonicalize(req(t, nil, sys, user1, asst1))
	cur := Canonicalize(req(t, nil, sys, user1))
	_, br := CommonPrefix(prev, cur)
	if br == nil || br.Reason != "removed" {
		t.Errorf("break = %+v", br)
	}
}

func TestCanonicalizeMultipartAndMalformed(t *testing.T) {
	multi := `{"role":"user","content":[{"type":"text","text":"look at this"},{"type":"image_url","image_url":{"url":"data:..."}}]}`
	p := Canonicalize(req(t, nil, multi))
	if !strings.Contains(p.Canonical, "user:look at this[image_url]") {
		t.Errorf("canonical = %q", p.Canonical)
	}
	bad := Canonicalize(req(t, nil, `{"role":42}`))
	if len(bad.Segments) != 1 || bad.Segments[0].Role != "?" {
		t.Errorf("malformed message segments = %+v", bad.Segments)
	}
}

func TestParseRequestErrors(t *testing.T) {
	if _, err := ParseRequest([]byte(`not json`)); err == nil {
		t.Error("expected error")
	}
	if _, err := ParseRequest([]byte(`{"model":"m"}`)); err == nil {
		t.Error("no messages should error")
	}
}

func TestRedact(t *testing.T) {
	in := "key sk-abcdefghijklmnop and ghp_ABCDEFGH1234 and Bearer eyJhbGciOi.xyz and AKIAABCDEFGHIJKL12"
	out := Redact(in)
	for _, leaked := range []string{"sk-abcdefghijklmnop", "ghp_ABCDEFGH1234", "eyJhbGciOi.xyz", "AKIAABCDEFGHIJKL12"} {
		if strings.Contains(out, leaked) {
			t.Errorf("leaked %q in %q", leaked, out)
		}
	}
	if !strings.Contains(out, "sk-a…[redacted]") {
		t.Errorf("redaction shape: %q", out)
	}
	if Redact("plain text") != "plain text" {
		t.Error("plain text should be untouched")
	}
	pem := "config:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0Z3VS5JJcds3xfn\n-----END RSA PRIVATE KEY-----\nnext"
	if out := Redact(pem); strings.Contains(out, "MIIEow") || !strings.Contains(out, "[redacted]") || !strings.HasSuffix(out, "next") {
		t.Errorf("PEM body leaked: %q", out)
	}
	jwt := "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
	if out := Redact(jwt); strings.Contains(out, "SflKxw") {
		t.Errorf("JWT leaked: %q", out)
	}
}

func TestWindowRedactsAcrossTheCropBoundary(t *testing.T) {
	// The secret starts inside the excerpt window but extends past it; the
	// excerpt must not show the leading fragment of the key.
	secret := "sk-" + strings.Repeat("A", 100)
	s := strings.Repeat("a", 30) + "key=" + secret + strings.Repeat("b", 300)
	w := window(s, 36) // inside "key=" just before the secret
	if strings.Contains(w, "sk-AAAA") {
		t.Errorf("credential fragment visible in excerpt: %q", w)
	}
	if !strings.Contains(w, "[redacted]") {
		t.Errorf("expected redaction marker in %q", w)
	}
}

func TestWindowClampsAndMarks(t *testing.T) {
	s := strings.Repeat("a", 10) + "\x00" + strings.Repeat("b", 200)
	w := window(s, 5)
	if !strings.HasPrefix(w, "aaaaa|aaaaa⏎") || len([]rune(w)) > ExcerptChars+3 {
		t.Errorf("window = %q", w)
	}
}
