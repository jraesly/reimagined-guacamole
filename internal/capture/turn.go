package capture

import "time"

// Turn is everything probe learned about one chat-completions request.
// Every numeric field carries its Source; see model.Source for the labels.
type Turn struct {
	Seq        int       `json:"seq"`
	Session    string    `json:"session"` // client key: user-agent + model
	At         time.Time `json:"at"`
	Model      string    `json:"model"`
	Stream     bool      `json:"stream"`
	Status     string    `json:"status"` // "ok", "upstream-error", "aborted", "unaccounted"
	HTTPStatus int       `json:"http_status"`
	Error      string    `json:"error,omitempty"`

	RequestBytes int         `json:"request_bytes"`
	MessageCount int         `json:"message_count"`
	ToolCount    int         `json:"tool_count"`
	Chars        Chars       `json:"chars"`
	Tools        []ToolChars `json:"tools,omitempty"`

	// PromptTokens is measured when the upstream reported usage, otherwise
	// inferred from characters. The split below is always inferred: it
	// apportions the total by characters.
	PromptTokens  Value        `json:"prompt_tokens"`
	UsageInjected bool         `json:"usage_injected"` // probe added stream_options.include_usage
	SystemTokens  Value        `json:"system_tokens"`
	ToolsTokens   Value        `json:"tools_tokens"`
	HistoryTokens Value        `json:"history_tokens"`
	CurrentTokens Value        `json:"current_tokens"`
	HarnessTokens Value        `json:"harness_tokens"` // system + tools: what the harness added
	PerTool       []ToolTokens `json:"per_tool,omitempty"`

	// Prefix stability against the previous turn in the same session.
	FirstTurn         bool   `json:"first_turn"`
	PrefixBytes       int    `json:"prefix_bytes"`       // canonical bytes shared with the previous turn
	ReprefilledTokens Value  `json:"reprefilled_tokens"` // inferred: prompt tokens beyond the shared prefix
	PrefixBreak       *Break `json:"prefix_break,omitempty"`

	// Timing observed by probe's clock. TTFT is the first generated token of
	// any kind; for thinking models that is a reasoning token, so TTFC (first
	// visible content or tool call) is what the user actually waits for.
	TTFTms       Value  `json:"ttft_ms"`
	TTFCms       *Value `json:"ttfc_ms,omitempty"` // absent when no content ever arrived
	DecodeMs     Value  `json:"decode_ms"`
	TotalMs      Value  `json:"total_ms"`
	PrefillShare Value  `json:"prefill_share"`           // ttft / total
	OutputTokens *Value `json:"output_tokens,omitempty"` // measured from usage when present
	DecodeTokS   *Value `json:"decode_tok_s,omitempty"`  // observed: output tokens / decode time

	Backend *Backend `json:"backend,omitempty"`
}

// ToolTokens is the inferred token cost of one tool definition.
type ToolTokens struct {
	Name   string `json:"name"`
	Tokens Value  `json:"tokens"`
}

// Backend is telemetry the server reported itself (all measured).
type Backend struct {
	Source           string   `json:"source"` // "llama.cpp timings" or "lmstudio stats"
	PromptN          *float64 `json:"prompt_n,omitempty"`
	PromptMs         *float64 `json:"prompt_ms,omitempty"`
	PredictedN       *float64 `json:"predicted_n,omitempty"`
	PredictedMs      *float64 `json:"predicted_ms,omitempty"`
	CacheN           *float64 `json:"cache_n,omitempty"` // prompt tokens served from the KV cache
	TokensPerSecond  *float64 `json:"tokens_per_second,omitempty"`
	TimeToFirstToken *float64 `json:"time_to_first_token,omitempty"`
}

// backendFrom converts parsed server fields into a Backend, or nil.
func backendFrom(o *Observation) *Backend {
	switch {
	case o.Timings != nil:
		t := o.Timings
		return &Backend{Source: "llama.cpp timings", PromptN: t.PromptN, PromptMs: t.PromptMs,
			PredictedN: t.PredictedN, PredictedMs: t.PredictedMs, CacheN: t.CacheN}
	case o.Stats != nil:
		return &Backend{Source: "lmstudio stats", TokensPerSecond: o.Stats.TokensPerSecond, TimeToFirstToken: o.Stats.TimeToFirstToken}
	}
	return nil
}
