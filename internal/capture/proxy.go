package capture

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/model"
)

const (
	defaultMaxBody   = 32 << 20 // 32 MiB
	maxRespBuffer    = 64 << 20 // 64 MiB: cap on in-memory buffering for non-streaming parsing and dumps
	streamChunkSize  = 32 << 10 // 32 KiB: relay read/flush granularity
	errorExcerptSize = 200      // bytes of an upstream error body kept in Turn.Error
)

// Options configures a Proxy.
type Options struct {
	Upstream *url.URL         // base such as http://localhost:1234/v1 or http://localhost:11434
	MaxBody  int64            // default 32 MiB when zero
	DumpDir  string           // when set, write redacted raw request/response bodies per turn
	OnTurn   func(Turn)       // called once per completed turn, any status
	Now      func() time.Time // defaults to time.Now; tests inject a clock for Turn.At
}

// Proxy is a session-scoped reverse proxy that forwards OpenAI-compatible
// chat-completions traffic byte-exact to Upstream while measuring what the
// harness added and how it decoded. Everything else is a plain pass-through.
type Proxy struct {
	opts   Options
	client *http.Client
	runID  string // distinguishes dump files of different sessions

	inflight sync.WaitGroup
	mu       sync.Mutex
	seq      int
	turns    []Turn
	sessions map[string]*Prompt // last prompt per session key, for prefix comparison
}

// SessionHeader lets a harness that runs parallel agents under one
// User-Agent keep their prefix histories apart.
const SessionHeader = "X-Probe-Session"

// New builds a Proxy. Upstream is required.
func New(opts Options) (*Proxy, error) {
	if opts.Upstream == nil {
		return nil, errors.New("capture: Options.Upstream is required")
	}
	if opts.MaxBody == 0 {
		opts.MaxBody = defaultMaxBody
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	// Never negotiate gzip on the harness's behalf: the bytes we relay must
	// be the bytes the upstream produced.
	transport.DisableCompression = true
	return &Proxy{
		opts: opts,
		client: &http.Client{ // zero Timeout: streaming responses may run indefinitely
			Transport: transport,
			// A redirect must reach the client as a redirect, not be followed
			// into a different resource (or a different method).
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		runID:    opts.Now().Format("20060102-150405"),
		sessions: map[string]*Prompt{},
	}, nil
}

// Wait blocks until every in-flight request has finished accounting or the
// timeout passes; it reports whether all of them finished.
func (p *Proxy) Wait(timeout time.Duration) bool {
	done := make(chan struct{})
	go func() { p.inflight.Wait(); close(done) }()
	select {
	case <-done:
		return true
	case <-time.After(timeout):
		return false
	}
}

// abortResponse drops the client connection so a truncated upstream body is
// not delivered with clean HTTP framing. net/http recovers ErrAbortHandler
// without logging. It must run after the turn has been recorded.
func abortResponse() { panic(http.ErrAbortHandler) }

// Turns returns a copy of the completed turns recorded so far.
func (p *Proxy) Turns() []Turn {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Turn, len(p.turns))
	copy(out, p.turns)
	return out
}

func (p *Proxy) nextSeq() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.seq++
	return p.seq
}

func (p *Proxy) addTurn(t Turn) {
	p.mu.Lock()
	p.turns = append(p.turns, t)
	p.mu.Unlock()
	if p.opts.OnTurn != nil {
		p.opts.OnTurn(t)
	}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.inflight.Add(1)
	defer p.inflight.Done()
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		p.passthrough(w, r)
		return
	}
	p.handleChatCompletions(w, r)
}

// hopByHopHeaders are stripped in both directions, along with any header
// the Connection header nominates (RFC 9110 §7.6.1).
var hopByHopHeaders = []string{"Connection", "Keep-Alive", "TE", "Trailer", "Transfer-Encoding", "Upgrade", "Proxy-Connection", "Proxy-Authenticate", "Proxy-Authorization"}

func stripHopByHop(h http.Header) {
	for _, v := range h.Values("Connection") {
		for _, name := range strings.Split(v, ",") {
			if name = strings.TrimSpace(name); name != "" {
				h.Del(name)
			}
		}
	}
	for _, k := range hopByHopHeaders {
		h.Del(k)
	}
}

// prepareOutHeader clones src for the outgoing request: hop-by-hop headers
// stripped and Accept-Encoding removed. With compression disabled on the
// transport the upstream then sends identity bytes, which are the bytes the
// client receives and the bytes the scanner reads.
func prepareOutHeader(src http.Header) http.Header {
	h := src.Clone()
	stripHopByHop(h)
	h.Del("Accept-Encoding")
	h.Del(SessionHeader)
	return h
}

// copyResponseHeaders copies resp's headers to w minus hop-by-hop headers.
// When the transport decompressed the body for us, Content-Encoding and
// Content-Length no longer describe what we are about to write.
func copyResponseHeaders(w http.ResponseWriter, resp *http.Response) {
	h := resp.Header.Clone()
	stripHopByHop(h)
	if resp.Uncompressed {
		h.Del("Content-Encoding")
		h.Del("Content-Length")
	}
	dst := w.Header()
	for k, vv := range h {
		dst[k] = vv
	}
}

// targetURL applies the path rule: Upstream.Scheme://Upstream.Host + join of
// Upstream.Path and the request path, with any Upstream.Path prefix already
// present on the request path stripped first so it is not duplicated.
func targetURL(upstream *url.URL, r *http.Request) *url.URL {
	// Work on the escaped form so /a%2Fb stays /a%2Fb.
	reqPath := r.URL.EscapedPath()
	prefix := strings.TrimSuffix(upstream.EscapedPath(), "/")
	remainder := reqPath
	if prefix != "" && strings.HasPrefix(reqPath, prefix+"/") {
		remainder = reqPath[len(prefix):]
	}
	u := *upstream
	joined := prefix + "/" + strings.TrimPrefix(remainder, "/")
	u.RawPath = joined
	if unescaped, err := url.PathUnescape(joined); err == nil {
		u.Path = unescaped
	} else {
		u.Path = joined
	}
	u.RawQuery = r.URL.RawQuery
	u.Fragment = ""
	return &u
}

// relay copies body to w in chunks, flushing after each write. Every sink
// sees each chunk before it is written so observations are stamped on
// arrival, not after a possibly slow client accepts the bytes. It never
// alters the bytes the client sees. It returns nil on a clean EOF, or the
// error (read or write) that stopped the copy.
func relay(w http.ResponseWriter, body io.Reader, sinks ...func([]byte)) error {
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, streamChunkSize)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			for _, sink := range sinks {
				sink(chunk)
			}
			if _, werr := w.Write(chunk); werr != nil {
				return werr
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			if rerr == io.EOF {
				return nil
			}
			return rerr
		}
	}
}

// startResponse sends status and headers immediately so a client is not
// left waiting for headers until the first body chunk arrives.
func startResponse(w http.ResponseWriter, resp *http.Response) {
	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func (p *Proxy) passthrough(w http.ResponseWriter, r *http.Request) {
	target := targetURL(p.opts.Upstream, r)
	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	outReq.Header = prepareOutHeader(r.Header)
	outReq.ContentLength = r.ContentLength

	resp, err := p.client.Do(outReq)
	if err != nil {
		http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	startResponse(w, resp)
	if err := relay(w, resp.Body); err != nil && r.Context().Err() == nil {
		abortResponse()
	}
}

// readCapped reads up to max+1 bytes of r. oversize is true when the body is
// longer than max; buf then holds only the first max+1 bytes actually read.
func readCapped(r io.Reader, max int64) (buf []byte, oversize bool, err error) {
	buf, err = io.ReadAll(io.LimitReader(r, max+1))
	if err != nil {
		return nil, false, err
	}
	return buf, int64(len(buf)) > max, nil
}

// needsInjection reports whether the request streams without asking for
// usage in its stream chunks.
func needsInjection(req *Request) bool {
	if !req.Stream {
		return false
	}
	_, ok := req.StreamOptions["include_usage"]
	return !ok
}

// injectUsage adds stream_options.include_usage=true to a request body.
// When the request has no stream_options key at all (the common case) the
// member is spliced in before the closing brace, so every other byte of the
// body is preserved verbatim. Only when a stream_options object already
// exists is the body re-marshalled, which can reorder keys. ok is false if
// raw is not a JSON object.
func injectUsage(raw []byte) (out []byte, ok bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, false
	}
	existing, has := obj["stream_options"]
	if !has || string(existing) == "null" {
		if has {
			// A literal null cannot be spliced around; rebuild without it.
			delete(obj, "stream_options")
			trimmed, err := json.Marshal(obj)
			if err != nil {
				return raw, false
			}
			raw = trimmed
		}
		end := bytes.LastIndexByte(raw, '}')
		if end < 0 {
			return raw, false
		}
		body := make([]byte, 0, len(raw)+40)
		body = append(body, bytes.TrimRight(raw[:end], " \t\r\n")...)
		body = append(body, `,"stream_options":{"include_usage":true}`...)
		body = append(body, raw[end:]...)
		return body, true
	}
	so := map[string]json.RawMessage{}
	if err := json.Unmarshal(existing, &so); err != nil || so == nil {
		so = map[string]json.RawMessage{}
	}
	so["include_usage"] = json.RawMessage("true")
	soBytes, err := json.Marshal(so)
	if err != nil {
		return raw, false
	}
	obj["stream_options"] = soBytes
	body, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return body, true
}

// newDumpSink returns a buffer and a sink that fills it up to maxRespBuffer,
// or a no-op sink when dumping is disabled.
func (p *Proxy) newDumpSink() (*bytes.Buffer, func([]byte)) {
	if p.opts.DumpDir == "" {
		return nil, func([]byte) {}
	}
	buf := &bytes.Buffer{}
	return buf, func(chunk []byte) {
		if buf.Len() >= maxRespBuffer {
			return
		}
		if remaining := maxRespBuffer - buf.Len(); len(chunk) > remaining {
			chunk = chunk[:remaining]
		}
		buf.Write(chunk)
	}
}

// dump writes a redacted copy as a private, new file. Names carry the run
// id so sessions never overwrite each other, and O_EXCL|O_NOFOLLOW refuses
// to write through an existing file or symlink.
func (p *Proxy) dump(name string, body []byte) {
	if p.opts.DumpDir == "" || body == nil {
		return
	}
	if err := os.MkdirAll(p.opts.DumpDir, 0o700); err != nil {
		return
	}
	path := filepath.Join(p.opts.DumpDir, p.runID+"-"+name)
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL|syscall.O_NOFOLLOW, 0o600)
	if err != nil {
		return
	}
	_, _ = f.WriteString(Redact(string(body)))
	_ = f.Close()
}

// handleChatCompletions implements rules 3-11 for POST .../chat/completions.
func (p *Proxy) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	seq := p.nextSeq()
	at := p.opts.Now()
	monoStart := time.Now() // wall-clock durations are measured separately from the injectable At clock

	raw, oversize, err := readCapped(r.Body, p.opts.MaxBody)
	if err != nil {
		total := time.Since(monoStart)
		turn := Turn{Seq: seq, Session: r.Header.Get("User-Agent") + "|", At: at, Status: "unaccounted",
			Error: err.Error(), HTTPStatus: http.StatusBadRequest}
		p.finalizeTiming(&turn, total, total)
		p.addTurn(turn)
		http.Error(w, "reading request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	preq, perr := ParseRequest(raw)
	if oversize || perr != nil {
		p.forwardUnaccounted(w, r, seq, at, monoStart, raw)
		return
	}

	prompt := Canonicalize(preq)
	chars, toolChars := CountChars(prompt)
	session := r.Header.Get("User-Agent") + "|" + preq.Model
	if s := r.Header.Get(SessionHeader); s != "" {
		session = s + "|" + preq.Model
	}

	p.mu.Lock()
	prev := p.sessions[session]
	p.sessions[session] = prompt
	p.mu.Unlock()

	firstTurn := prev == nil
	var prefixBytes int
	var prefixBreak *Break
	if !firstTurn {
		prefixBytes, prefixBreak = CommonPrefix(prev, prompt)
	}

	body, injected := raw, false
	if needsInjection(preq) {
		if out, ok := injectUsage(raw); ok {
			body, injected = out, true
		}
	}

	turn := Turn{
		Seq: seq, Session: session, At: at, Model: preq.Model, Stream: preq.Stream,
		RequestBytes: len(raw), MessageCount: len(preq.Messages), ToolCount: len(preq.Tools),
		Chars: chars, Tools: toolChars, UsageInjected: injected,
		FirstTurn: firstTurn, PrefixBytes: prefixBytes, PrefixBreak: prefixBreak,
	}

	target := targetURL(p.opts.Upstream, r)
	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		total := time.Since(monoStart)
		turn.Status, turn.Error, turn.HTTPStatus = "unaccounted", err.Error(), http.StatusBadGateway
		p.finalizeTiming(&turn, total, total)
		p.dump(fmt.Sprintf("turn-%d-request.json", seq), raw)
		p.addTurn(turn)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	outReq.Header = prepareOutHeader(r.Header)
	outReq.ContentLength = int64(len(body))
	outReq.Header.Set("Content-Length", strconv.Itoa(len(body)))

	resp, err := p.client.Do(outReq)
	if err != nil {
		total := time.Since(monoStart)
		turn.HTTPStatus = http.StatusBadGateway
		if r.Context().Err() != nil {
			turn.Status = "aborted"
		} else {
			turn.Status, turn.Error = "upstream-error", Redact(err.Error())
		}
		p.finalizeTiming(&turn, total, total)
		p.dump(fmt.Sprintf("turn-%d-request.json", seq), raw)
		p.addTurn(turn)
		http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	startResponse(w, resp)
	turn.HTTPStatus = resp.StatusCode

	isSSE := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") || preq.Stream

	obs := &Observation{}
	var ttft, ttfc time.Duration
	scanner := &sseScanner{obs: obs,
		onFirst:        func() { ttft = time.Since(monoStart) },
		onFirstContent: func() { ttfc = time.Since(monoStart) },
	}

	var head []byte
	captureHead := func(chunk []byte) {
		if len(head) >= errorExcerptSize {
			return
		}
		need := errorExcerptSize - len(head)
		if need > len(chunk) {
			need = len(chunk)
		}
		head = append(head, chunk[:need]...)
	}

	var respBuf bytes.Buffer
	bufferBody := func(chunk []byte) {
		if respBuf.Len() >= maxRespBuffer {
			return
		}
		if remaining := maxRespBuffer - respBuf.Len(); len(chunk) > remaining {
			chunk = chunk[:remaining]
		}
		respBuf.Write(chunk)
	}

	dumpBuf, dumpSink := p.newDumpSink()
	sinks := []func([]byte){captureHead, dumpSink}
	if isSSE {
		sinks = append(sinks, func(b []byte) { _, _ = scanner.Write(b) })
	} else {
		sinks = append(sinks, bufferBody)
	}

	relayErr := relay(w, resp.Body, sinks...)
	total := time.Since(monoStart)
	if ttft == 0 {
		ttft = total // non-streaming, or a stream that never emitted content
	}
	if !isSSE {
		parseBody(respBuf.Bytes(), obs)
		if obs.FirstContent {
			ttfc = total
		}
	}
	if obs.FirstContent && ttfc > 0 {
		v := Value{float64(ttfc.Milliseconds()), model.Observed}
		turn.TTFCms = &v
	}

	switch {
	case relayErr != nil && r.Context().Err() != nil:
		turn.Status = "aborted"
	case relayErr != nil:
		turn.Status, turn.Error = "upstream-error", Redact(relayErr.Error())
	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		turn.Status, turn.Error = "upstream-error", Redact(string(head))
	default:
		turn.Status = "ok"
	}

	p.finishAccounting(&turn, prompt, chars, toolChars, obs, ttft, total, prefixBytes, firstTurn)

	p.dump(fmt.Sprintf("turn-%d-request.json", seq), raw)
	if dumpBuf != nil {
		p.dump(fmt.Sprintf("turn-%d-response.txt", seq), dumpBuf.Bytes())
	}
	p.addTurn(turn)
	if relayErr != nil && r.Context().Err() == nil {
		abortResponse()
	}
}

// forwardUnaccounted handles a body that exceeded MaxBody or did not parse:
// it is forwarded exactly as received (buffered bytes plus whatever of
// r.Body was not yet read), timed, but not accounted for.
func (p *Proxy) forwardUnaccounted(w http.ResponseWriter, r *http.Request, seq int, at time.Time, monoStart time.Time, buffered []byte) {
	turn := Turn{Seq: seq, Session: r.Header.Get("User-Agent") + "|", At: at, Status: "unaccounted", RequestBytes: len(buffered)}

	target := targetURL(p.opts.Upstream, r)
	bodyReader := io.MultiReader(bytes.NewReader(buffered), r.Body)
	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bodyReader)
	if err != nil {
		total := time.Since(monoStart)
		turn.Error, turn.HTTPStatus = err.Error(), http.StatusBadGateway
		p.finalizeTiming(&turn, total, total)
		p.dump(fmt.Sprintf("turn-%d-request.json", seq), buffered)
		p.addTurn(turn)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	outReq.Header = prepareOutHeader(r.Header)
	outReq.ContentLength = r.ContentLength

	resp, err := p.client.Do(outReq)
	if err != nil {
		total := time.Since(monoStart)
		turn.Error, turn.HTTPStatus = Redact(err.Error()), http.StatusBadGateway
		p.finalizeTiming(&turn, total, total)
		p.dump(fmt.Sprintf("turn-%d-request.json", seq), buffered)
		p.addTurn(turn)
		http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	startResponse(w, resp)
	turn.HTTPStatus = resp.StatusCode

	dumpBuf, dumpSink := p.newDumpSink()
	relayErr := relay(w, resp.Body, dumpSink)
	total := time.Since(monoStart)
	if relayErr != nil {
		turn.Error = Redact(relayErr.Error())
	}
	p.finalizeTiming(&turn, total, total)

	p.dump(fmt.Sprintf("turn-%d-request.json", seq), buffered)
	if dumpBuf != nil {
		p.dump(fmt.Sprintf("turn-%d-response.txt", seq), dumpBuf.Bytes())
	}
	p.addTurn(turn)
	if relayErr != nil && r.Context().Err() == nil {
		abortResponse()
	}
}

// finalizeTiming fills the observed timing fields shared by every path.
func (p *Proxy) finalizeTiming(turn *Turn, ttft, total time.Duration) {
	ttftMs, totalMs := float64(ttft.Milliseconds()), float64(total.Milliseconds())
	turn.TTFTms = Value{ttftMs, model.Observed}
	turn.TotalMs = Value{totalMs, model.Observed}
	turn.DecodeMs = Value{totalMs - ttftMs, model.Observed}
	share := 0.0
	if totalMs > 0 {
		share = ttftMs / totalMs
	}
	turn.PrefillShare = Value{share, model.Observed}
}

// finishAccounting fills in token accounting (rules 6-8) once the response
// is known: promptTokens (measured or inferred), the character-apportioned
// splits (always inferred), prefix reuse, and decode-side output tokens.
func (p *Proxy) finishAccounting(turn *Turn, cur *Prompt, chars Chars, toolChars []ToolChars, obs *Observation, ttft, total time.Duration, prefixBytes int, firstTurn bool) {
	turn.Backend = backendFrom(obs)

	var promptTokens float64
	if obs.Usage != nil {
		promptTokens = float64(obs.Usage.PromptTokens)
		turn.PromptTokens = Value{promptTokens, model.Measured}
	} else {
		promptTokens = EstimateTokens(chars.Total)
		turn.PromptTokens = Value{promptTokens, model.Inferred}
	}

	turn.SystemTokens = Value{Apportion(promptTokens, chars.System, chars.Total), model.Inferred}
	turn.ToolsTokens = Value{Apportion(promptTokens, chars.Tools, chars.Total), model.Inferred}
	turn.HistoryTokens = Value{Apportion(promptTokens, chars.History, chars.Total), model.Inferred}
	turn.CurrentTokens = Value{Apportion(promptTokens, chars.Current, chars.Total), model.Inferred}
	turn.HarnessTokens = Value{turn.SystemTokens.Value + turn.ToolsTokens.Value, model.Inferred}

	for _, tc := range toolChars {
		turn.PerTool = append(turn.PerTool, ToolTokens{
			Name:   tc.Name,
			Tokens: Value{Apportion(promptTokens, tc.Chars, chars.Total), model.Inferred},
		})
	}

	if firstTurn {
		turn.ReprefilledTokens = Value{promptTokens, model.Inferred}
	} else {
		curLen := len(cur.Canonical)
		turn.ReprefilledTokens = Value{Apportion(promptTokens, curLen-prefixBytes, curLen), model.Inferred}
	}

	p.finalizeTiming(turn, ttft, total)

	if obs.Usage != nil {
		out := Value{float64(obs.Usage.CompletionTokens), model.Measured}
		turn.OutputTokens = &out
		if decodeSeconds := (total - ttft).Seconds(); decodeSeconds > 0 {
			turn.DecodeTokS = &Value{out.Value / decodeSeconds, model.Observed}
		}
	}
}
