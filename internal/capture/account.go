// Package capture is a session-scoped OpenAI-compatible proxy that measures
// what a coding harness sends to a local model server: how many tokens the
// harness added, how much of the prompt prefix survived from the previous
// turn, and where the latency went.
package capture

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

// Request is the subset of a chat-completions body that accounting needs.
// Raw fields are kept so the request is forwarded exactly as received.
type Request struct {
	Model         string            `json:"model"`
	Stream        bool              `json:"stream"`
	StreamOptions map[string]any    `json:"stream_options"`
	Messages      []json.RawMessage `json:"messages"`
	Tools         []json.RawMessage `json:"tools"`
}

// Segment is one span of the canonical prompt with its origin.
type Segment struct {
	Kind   string // "system", "tools", "message"
	Index  int    // message index, or tool index for kind "tools"
	Role   string
	Field  string // "content", "tool_calls", "tool" (tool definition)
	Name   string // tool name for kind "tools"
	Start  int    // byte offset in the canonical string
	End    int
	Length int
}

// Prompt is the canonical, order-preserving serialization of a request used
// for prefix comparison and character accounting.
type Prompt struct {
	Canonical string
	Segments  []Segment
	Model     string
	Stream    bool
}

// Chars are character counts per part of the prompt.
type Chars struct {
	System  int `json:"system"`
	Tools   int `json:"tools"`
	History int `json:"history"`
	Current int `json:"current"`
	Total   int `json:"total"`
}

// ToolChars is the size of one tool definition.
type ToolChars struct {
	Name  string `json:"name"`
	Chars int    `json:"chars"`
}

// ParseRequest decodes a chat-completions body.
func ParseRequest(body []byte) (*Request, error) {
	var r Request
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, err
	}
	if len(r.Messages) == 0 {
		return nil, fmt.Errorf("no messages")
	}
	return &r, nil
}

type message struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	ToolCalls json.RawMessage `json:"tool_calls"`
}

type toolDef struct {
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

// Canonicalize serializes tools then messages in order. Tools are placed
// first because chat templates inject them ahead of the conversation; the
// exact template is server-specific, so prefix results are labeled inferred.
func Canonicalize(r *Request) *Prompt {
	var b strings.Builder
	p := &Prompt{Model: r.Model, Stream: r.Stream}
	add := func(seg Segment, text string) {
		seg.Start = b.Len()
		b.WriteString(text)
		b.WriteByte(0)
		seg.End = b.Len()
		seg.Length = len(text)
		p.Segments = append(p.Segments, seg)
	}
	for i, raw := range r.Tools {
		var t toolDef
		_ = json.Unmarshal(raw, &t)
		add(Segment{Kind: "tools", Index: i, Field: "tool", Name: t.Function.Name}, compactJSON(raw))
	}
	for i, raw := range r.Messages {
		var m message
		if json.Unmarshal(raw, &m) != nil {
			add(Segment{Kind: "message", Index: i, Role: "?", Field: "content"}, compactJSON(raw))
			continue
		}
		kind := "message"
		if m.Role == "system" || m.Role == "developer" {
			kind = "system"
		}
		add(Segment{Kind: kind, Index: i, Role: m.Role, Field: "content"}, m.Role+":"+contentText(m.Content))
		if len(m.ToolCalls) > 0 && string(m.ToolCalls) != "null" {
			add(Segment{Kind: kind, Index: i, Role: m.Role, Field: "tool_calls"}, compactJSON(m.ToolCalls))
		}
	}
	p.Canonical = b.String()
	return p
}

// contentText flattens string or multi-part content to text.
func contentText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				b.WriteString(p.Text)
			} else {
				b.WriteByte('[')
				b.WriteString(p.Type)
				b.WriteByte(']')
			}
		}
		return b.String()
	}
	return compactJSON(raw)
}

func compactJSON(raw json.RawMessage) string {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	out, err := json.Marshal(v) // map keys are sorted by encoding/json
	if err != nil {
		return string(raw)
	}
	return string(out)
}

// CountChars splits the prompt into harness-added and conversation parts.
// The last user message is "current"; every other non-system message is
// "history".
func CountChars(p *Prompt) (Chars, []ToolChars) {
	var c Chars
	var tools []ToolChars
	lastUser := -1
	for _, s := range p.Segments {
		if s.Kind == "message" && s.Role == "user" {
			lastUser = s.Index
		}
	}
	for _, s := range p.Segments {
		switch {
		case s.Kind == "tools":
			c.Tools += s.Length
			tools = append(tools, ToolChars{Name: s.Name, Chars: s.Length})
		case s.Kind == "system":
			c.System += s.Length
		case s.Role == "user" && s.Index == lastUser:
			c.Current += s.Length
		default:
			c.History += s.Length
		}
	}
	c.Total = c.System + c.Tools + c.History + c.Current
	sort.Slice(tools, func(i, j int) bool { return tools[i].Chars > tools[j].Chars })
	return c, tools
}

// Apportion converts a character count into tokens using the server-reported
// prompt total: tokens = promptTokens × chars / totalChars. The result is
// inferred even though the total is measured.
func Apportion(promptTokens float64, chars, totalChars int) float64 {
	if totalChars == 0 {
		return 0
	}
	return promptTokens * float64(chars) / float64(totalChars)
}

// EstimateTokens is the fallback when the server reports no usage: roughly
// four characters per token for English and code.
func EstimateTokens(chars int) float64 { return float64(chars) / 4 }

// Break describes where two consecutive prompts diverge.
type Break struct {
	Segment string `json:"segment"` // e.g. "system[0].content", "tools[3] (read_file)", "message[7].tool_calls"
	Offset  int    `json:"offset"`  // character offset within the segment
	Excerpt string `json:"excerpt"` // redacted window around the divergence, "|" marks it
	Reason  string `json:"reason"`  // "changed", "appended", "removed", "reordered"
}

// CommonPrefix compares prev and cur and returns the shared byte length and
// the first divergence. A prompt that only appends to the previous one has
// no break: that is the healthy case.
func CommonPrefix(prev, cur *Prompt) (int, *Break) {
	a, b := prev.Canonical, cur.Canonical
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	if n == len(a) {
		return n, nil // cur extends prev, or is identical
	}
	seg := segmentAt(cur, n)
	if seg == nil {
		return n, &Break{Segment: "end", Reason: "removed", Excerpt: window(a, n)}
	}
	br := &Break{Segment: seg.label(), Offset: n - seg.Start, Reason: "changed", Excerpt: window(b, n)}
	if n == len(b) {
		br.Reason = "removed"
		br.Excerpt = window(a, n)
	} else if prevSeg := segmentAt(prev, n); prevSeg != nil && prevSeg.Kind == "tools" && seg.Kind == "tools" && prevSeg.Name != seg.Name {
		br.Reason = "reordered"
	}
	return n, br
}

func segmentAt(p *Prompt, off int) *Segment {
	for i := range p.Segments {
		s := &p.Segments[i]
		if off >= s.Start && off < s.End {
			return s
		}
	}
	return nil
}

func (s Segment) label() string {
	switch s.Kind {
	case "tools":
		return fmt.Sprintf("tools[%d] (%s)", s.Index, s.Name)
	default:
		return fmt.Sprintf("%s[%d].%s", s.Kind, s.Index, s.Field)
	}
}

// ExcerptChars is the total width of the window shown around a break.
const ExcerptChars = 80

// redactMargin is how far beyond the excerpt window redaction looks, so a
// credential straddling the crop boundary is masked whole rather than
// exposed as a fragment.
const redactMargin = 512

func window(s string, at int) string {
	half := ExcerptChars / 2
	wlo, whi := max(at-half-redactMargin, 0), min(at+half+redactMargin, len(s))
	wide, mark := redactAt(s[wlo:whi], at-wlo)
	lo, hi := max(mark-half, 0), min(mark+half, len(wide))
	out := wide[lo:mark] + "|" + wide[mark:hi]
	out = strings.ReplaceAll(out, "\x00", "⏎")
	out = strings.ReplaceAll(out, "\n", "⏎")
	return out
}

var (
	secretRe = regexp.MustCompile(`(?i)(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|gho_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AKIA[A-Z0-9]{12,}|bearer\s+[A-Za-z0-9._-]{8,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})`)
	pemRe    = regexp.MustCompile(`(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|\z)`)
)

// Redact masks common credential shapes in text meant for reports: API
// keys, GitHub and Slack tokens, AWS access keys, bearer tokens, JWTs and
// whole PEM private-key blocks.
func Redact(s string) string {
	out, _ := redactAt(s, -1)
	return out
}

// redactAt redacts s and returns where position pos ends up. A position
// inside a masked span moves to the end of the mask, so an excerpt marker
// can never split a credential and expose its prefix.
func redactAt(s string, pos int) (string, int) {
	s, pos = mask(s, pos, pemRe, func(string) string {
		return "-----BEGIN PRIVATE KEY-----[redacted]-----END PRIVATE KEY-----"
	})
	return mask(s, pos, secretRe, func(m string) string {
		if len(m) <= 8 {
			return "[redacted]"
		}
		return m[:4] + "…[redacted]"
	})
}

func mask(s string, pos int, re *regexp.Regexp, repl func(string) string) (string, int) {
	var b strings.Builder
	last, newPos := 0, pos
	for _, m := range re.FindAllStringIndex(s, -1) {
		b.WriteString(s[last:m[0]])
		r := repl(s[m[0]:m[1]])
		switch {
		case pos >= m[1]:
			newPos += len(r) - (m[1] - m[0])
		case pos > m[0]:
			newPos = b.Len() + len(r)
		}
		b.WriteString(r)
		last = m[1]
	}
	b.WriteString(s[last:])
	return b.String(), newPos
}

// Value is a number with provenance, mirroring the fit report.
type Value struct {
	Value  float64      `json:"value"`
	Source model.Source `json:"source"`
}
