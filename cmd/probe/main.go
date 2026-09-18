// Command probe diagnoses local coding-agent setups. v0.1 ships `fit`.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/calib"
	"github.com/jraesly/reimagined-guacamole/internal/fit"
	"github.com/jraesly/reimagined-guacamole/internal/gguf"
	"github.com/jraesly/reimagined-guacamole/internal/hw"
	"github.com/jraesly/reimagined-guacamole/internal/measure"
	"github.com/jraesly/reimagined-guacamole/internal/model"
	"github.com/jraesly/reimagined-guacamole/internal/remote"
	"github.com/jraesly/reimagined-guacamole/internal/report"
	"github.com/jraesly/reimagined-guacamole/internal/scan"
	"github.com/jraesly/reimagined-guacamole/internal/suggest"
)

var version = "0.1.0-dev"

func main() {
	if len(os.Args) < 2 {
		if isTerminal(os.Stdin) {
			if err := runInteractive(os.Stdin, os.Stdout); err != nil {
				fmt.Fprintln(os.Stderr, "probe:", err)
				os.Exit(1)
			}
			return
		}
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "fit":
		err = runFit(os.Args[2:])
	case "capture":
		err = runCapture(os.Args[2:])
	case "header":
		err = runHeader(os.Args[2:])
	case "version":
		fmt.Println("probe", version)
	case "-h", "--help", "help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "probe:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  probe                                            interactive menu (on a terminal)
  probe fit [flags] [target ...]                   fit models to this machine
  probe capture --upstream URL [flags]             proxy a harness and measure what it sends
  probe header model.gguf                          dump GGUF header metadata (arrays summarized)
  probe version

targets: a .gguf path, an MLX directory, ollama:<name>[:<tag>] or
hf:<owner>/<repo>[:<file-filter>]. Remote targets read only the model header
(a few MB); nothing is downloaded. With no target, fit scans LM Studio, Ollama
and the Hugging Face cache. --suggest --online fits the suggested models too.`)
}

func runFit(args []string) error {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	contexts := fs.String("context", "", "context sizes to evaluate, e.g. 32k or 8k,32k,128k (default 8k..128k)")
	kv := fs.String("kv", "f16", "KV cache type: f16, q8_0 or q4_0")
	reserve := fs.Float64("reserve-gb", -1, "memory kept for OS, harness and browser (default 8 unified, 2 discrete)")
	bandwidth := fs.Float64("bandwidth-gb-s", 0, "memory bandwidth override for the speed estimate")
	asJSON := fs.Bool("json", false, "emit JSON with source labels")
	doSuggest := fs.Bool("suggest", false, "append curated baseline models for this memory class")
	online := fs.Bool("online", false, "with --suggest, read each suggested model's header from its registry and fit it")
	forTask := fs.String("for", "", "keep only models for this task (coding, agent, chat, vision, embedding); also filters --suggest")
	doMeasure := fs.Bool("measure", false, "run each installed Ollama model briefly and record its measured decode speed for calibration")
	measureCtx := fs.Int("measure-ctx", 4096, "context size used for --measure runs")
	measureTokens := fs.Int("measure-tokens", 128, "tokens generated per --measure run")
	calibPath := fs.String("calibration", "", "calibration file (default ~/.config/probe/calibration.json)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	ctxs, err := parseContexts(*contexts)
	if err != nil {
		return err
	}
	kvType := fit.KVType(*kv)
	if _, err := kvType.BytesPerElement(); err != nil {
		return err
	}

	info, err := hw.Detect()
	if err != nil {
		return fmt.Errorf("hardware detection: %w", err)
	}
	if *bandwidth > 0 {
		info.BandwidthGBs, info.BandwidthKnown = *bandwidth, true
		info.Notes = append(info.Notes, "bandwidth supplied on the command line")
	}
	if *reserve < 0 {
		*reserve = hw.DefaultReserveGB(info)
	}
	budget := info.Budget(*reserve)
	opts := fit.Options{Contexts: ctxs, KV: kvType}

	if *calibPath == "" {
		if *calibPath, err = calib.DefaultPath(); err != nil {
			return err
		}
	}
	cal, err := calib.Load(*calibPath)
	if err != nil {
		return err
	}
	if cal.Chip == "" {
		cal.Chip = info.Chip
	}
	var ollamaAvailable, lmstudioAvailable bool
	if *doMeasure {
		checkCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		ollamaAvailable = measure.OllamaAvailable(checkCtx)
		cancel()
		checkCtx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
		lmstudioAvailable = measure.LMStudioAvailable(checkCtx)
		cancel()
		if !lmstudioAvailable {
			fmt.Fprintln(os.Stderr, "--measure: LM Studio API unavailable; run `lms server start`; skipping LM Studio targets")
		}
		if !ollamaAvailable && !lmstudioAvailable {
			return fmt.Errorf("--measure needs the Ollama API at %s; start Ollama or omit --measure", measure.OllamaBase)
		}
	}
	measureOne := func(m *model.Model, t scan.Found) bool {
		name := firstName(t)
		backend, label := "ollama", "Ollama"
		sampleAliases := aliases(t)
		mctx, cancel := context.WithTimeout(context.Background(), measure.Timeout)
		defer cancel()
		id := name
		if t.Source == "lmstudio" {
			if !lmstudioAvailable {
				return false
			}
			backend, label = "lmstudio", "LM Studio"
			owner, repo, ok := strings.Cut(name, "/")
			if !ok || owner == "" || repo == "" {
				fmt.Fprintf(os.Stderr, "--measure: skipping %s: missing LM Studio owner/repo name\n", name)
				return false
			}
			models, err := measure.LMStudioModels(mctx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "--measure: skipping %s: %v\n", name, err)
				return false
			}
			matched, ok := measure.MatchLMStudio(models, owner, repo)
			if !ok {
				fmt.Fprintf(os.Stderr, "--measure: skipping %s: no matching LM Studio generation model\n", name)
				return false
			}
			id = matched.ID
			sampleAliases = []string{id}
		} else if !ollamaAvailable {
			return false
		}
		fmt.Fprintf(os.Stderr, "measuring %s at %dk context, %d tokens (%s loads the model; this can take a minute)…", name, *measureCtx/1024, *measureTokens, label)
		var res measure.Result
		var err error
		if backend == "lmstudio" {
			res, err = measure.LMStudio(mctx, id, *measureCtx, *measureTokens, nil)
		} else {
			res, err = measure.Ollama(mctx, name, *measureCtx, *measureTokens)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, " failed: %v\n", err)
			return false
		}
		bpt, _ := m.BytesPerToken()
		cal.Add(calib.Sample{Model: name, Aliases: sampleAliases, Kind: fit.Kind(m), Backend: backend, TokPerSec: res.TokPerSec,
			PromptTokSec: res.PromptTokPerSec, BytesPerToken: bpt, Context: res.Context, OutputTokens: res.OutputTokens,
			MeasuredAt: time.Now().UTC()})
		if backend == "lmstudio" {
			fmt.Fprintf(os.Stderr, " %.1f tok/s decode (measured), TTFT %.2fs, generation %.2fs, runtime %s\n", res.TokPerSec, res.TTFTSeconds, res.TotalSeconds, res.Runtime)
			for _, note := range res.Notes {
				fmt.Fprintf(os.Stderr, "--measure: %s: %s\n", name, note)
			}
		} else {
			fmt.Fprintf(os.Stderr, " %.1f tok/s decode, %.0f tok/s prompt, load %.0fs\n", res.TokPerSec, res.PromptTokPerSec, res.LoadSeconds)
		}
		return true
	}

	var local []string
	var refs []remote.Ref
	for _, a := range fs.Args() {
		if r, ok := remote.ParseRef(a); ok {
			refs = append(refs, r)
		} else {
			local = append(local, a)
		}
	}
	var targets []scan.Found
	if len(local) > 0 || len(refs) == 0 {
		if targets, err = resolveTargets(local); err != nil {
			return err
		}
	}
	rep := report.Report{Hardware: info, Budget: budget, KV: kvType}
	ctx := context.Background()
	addRemote := func(r remote.Ref) {
		m, err := remote.Resolve(ctx, r)
		if err != nil {
			rep.Models = append(rep.Models, report.ModelResult{Name: r.String(), Path: r.String(), Error: err.Error()})
			return
		}
		rep.Models = append(rep.Models, report.Build(m, nil, nil, budget.GB, opts, info.BandwidthGBs, cal))
	}
	for _, r := range refs {
		addRemote(r)
	}
	// Load every local model first, measure second, report third, so a
	// measurement taken this run calibrates every model in the report.
	type loaded struct {
		found scan.Found
		m     *model.Model
		err   error
	}
	var locals []loaded
	for _, t := range targets {
		var m *model.Model
		var merr error
		if t.IsDir {
			m, merr = model.FromMLXDir(t.Path)
		} else {
			m, merr = model.FromGGUF(t.Path)
		}
		if merr == nil {
			if t.Owner != "" && (m.Owner == "" || t.Source == "ollama") {
				m.Owner = t.Owner
				m.Baseline, m.BaselineNote = model.Classify(m.Owner, m.Name)
			}
			if len(t.Names) > 0 {
				m.Name = t.Names[0]
			}
			if t.Vision {
				m.AddTask("vision")
			}
			if task := strings.ToLower(strings.TrimSpace(*forTask)); task != "" && !m.HasTask(task) {
				continue // not for this task; leave it out of the report
			}
		}
		locals = append(locals, loaded{t, m, merr})
	}
	measured := 0
	if *doMeasure {
		for _, l := range locals {
			if l.err == nil && (l.found.Source == "ollama" || l.found.Source == "lmstudio") {
				if measureOne(l.m, l.found) {
					measured++
				}
			}
		}
	}
	for _, l := range locals {
		if l.err != nil {
			rep.Models = append(rep.Models, report.ModelResult{Name: firstName(l.found), Path: l.found.Path, Error: l.err.Error()})
			continue
		}
		rep.Models = append(rep.Models, report.Build(l.m, aliases(l.found), l.found.Extras, budget.GB, opts, info.BandwidthGBs, cal))
	}
	if *doMeasure {
		if measured == 0 {
			fmt.Fprintln(os.Stderr, "--measure: no target models successfully measured")
		} else if err := cal.Save(*calibPath); err != nil {
			return fmt.Errorf("saving calibration: %w", err)
		} else {
			fmt.Fprintf(os.Stderr, "calibration saved to %s\n", *calibPath)
		}
	}
	if len(rep.Models) == 0 && !*doSuggest {
		if *forTask != "" {
			return fmt.Errorf("no installed models for %q; try --suggest --for %s", *forTask, *forTask)
		}
		return errors.New("no models found; pass a .gguf path or an MLX directory, or use --suggest")
	}

	if *doSuggest {
		list, err := suggest.Load()
		if err != nil {
			return err
		}
		if err := list.CheckFresh(time.Now()); err != nil {
			return err
		}
		mem := info.RAMGB
		if !info.Unified && info.VRAMGB > 0 {
			mem = info.VRAMGB
		}
		if c, ok := list.ForMemory(mem); ok {
			filtered, ok, why := c.Filter(*forTask)
			if !ok {
				return fmt.Errorf("--for %s: %s", *forTask, why)
			}
			c = filtered
			if *forTask != "" {
				c.Name += " for " + strings.ToLower(*forTask)
			}
			home, _ := os.UserHomeDir()
			hosts := detectHosts(home)
			c.Models = filterHosts(c.Models, hosts)
			rep.Suggestion, rep.ListDate, rep.ListNote = &c, list.Updated, list.Note
			if len(hosts) == 0 {
				rep.ListNote += " No model host detected (Ollama, LM Studio); showing every entry."
			}
			if *online {
				installed := map[string]bool{}
				for _, m := range rep.Models {
					installed[m.Name] = true
					for _, a := range m.Aliases {
						installed[a] = true
					}
				}
				for _, e := range c.Models {
					if installed[e.Name] {
						continue // already on disk and reported above
					}
					if r, ok := remote.ParseRef(e.Ref); ok {
						addRemote(r)
					}
				}
			}
		}
	}

	if *asJSON {
		return report.WriteJSON(os.Stdout, rep)
	}
	report.WriteText(os.Stdout, rep)
	return nil
}

func runHeader(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: probe header model.gguf")
	}
	h, err := gguf.ReadHeader(args[0])
	if err != nil {
		return err
	}
	keys := make([]string, 0, len(h.Metadata))
	for k := range h.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("version %d, %d tensors, %d metadata keys\n", h.Version, h.TensorCount, len(keys))
	for _, k := range keys {
		v := h.Metadata[k]
		if arr, ok := v.([]any); ok {
			if len(arr) > 16 {
				fmt.Printf("%s = [%d elements; first: %v]\n", k, len(arr), arr[:4])
			} else {
				fmt.Printf("%s = %v\n", k, arr)
			}
			continue
		}
		if s, ok := v.(string); ok && len(s) > 120 {
			v = s[:120] + "…"
		}
		fmt.Printf("%s = %v\n", k, v)
	}
	return nil
}

func resolveTargets(paths []string) ([]scan.Found, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return scan.All(home)
	}
	// An explicit path may be a file the scanners know by a better name
	// (an Ollama blob, an LM Studio repo); reuse that identity when it is.
	known := map[string]scan.Found{}
	if all, err := scan.All(home); err == nil {
		for _, f := range all {
			known[f.Path] = f
		}
	}
	var out []scan.Found
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		if abs, err := filepath.Abs(p); err == nil {
			if f, ok := known[abs]; ok {
				out = append(out, f)
				continue
			}
		}
		if st.IsDir() {
			if _, err := os.Stat(p + "/config.json"); err == nil {
				out = append(out, scan.Found{Path: p, IsDir: true})
				continue
			}
			found := scan.HFLayout(p, "path")
			if len(found) == 0 {
				return nil, fmt.Errorf("%s: no config.json and no <owner>/<repo>/*.gguf inside", p)
			}
			out = append(out, found...)
			continue
		}
		out = append(out, scan.Found{Path: p})
	}
	return out, nil
}

// detectHosts reports which model hosts this machine has: "ollama" when the
// CLI or its model store exists, "hf" when LM Studio or the Hugging Face
// cache does (both pull from Hugging Face).
func detectHosts(home string) map[string]bool {
	hosts := map[string]bool{}
	if _, err := exec.LookPath("ollama"); err == nil {
		hosts["ollama"] = true
	}
	roots := scan.Roots(home)
	for key, host := range map[string]string{"ollama": "ollama", "lmstudio": "hf", "lmstudio2": "hf", "hfcache": "hf"} {
		if st, err := os.Stat(roots[key]); err == nil && st.IsDir() {
			hosts[host] = true
		}
	}
	return hosts
}

// filterHosts keeps entries for detected hosts; with none detected it keeps all.
func filterHosts(entries []suggest.Entry, hosts map[string]bool) []suggest.Entry {
	if len(hosts) == 0 {
		return entries
	}
	var out []suggest.Entry
	for _, e := range entries {
		if hosts[e.Host] {
			out = append(out, e)
		}
	}
	return out
}

func firstName(f scan.Found) string {
	if len(f.Names) > 0 {
		return f.Names[0]
	}
	return f.Path
}

func aliases(f scan.Found) []string {
	if len(f.Names) > 1 {
		return f.Names[1:]
	}
	return nil
}

// parseContexts accepts "32k", "32768", or comma-separated lists of either.
func parseContexts(s string) ([]uint64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var out []uint64
	for _, part := range strings.Split(s, ",") {
		p := strings.ToLower(strings.TrimSpace(part))
		mult := uint64(1)
		if q, ok := strings.CutSuffix(p, "k"); ok {
			mult, p = 1024, q
		}
		n, err := strconv.ParseUint(p, 10, 64)
		if err != nil || n == 0 {
			return nil, fmt.Errorf("bad context size %q (use e.g. 32k or 32768)", part)
		}
		out = append(out, n*mult)
	}
	return out, nil
}
