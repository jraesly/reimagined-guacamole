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

	"github.com/jraesly/reimagined-guacamole/internal/fit"
	"github.com/jraesly/reimagined-guacamole/internal/gguf"
	"github.com/jraesly/reimagined-guacamole/internal/hw"
	"github.com/jraesly/reimagined-guacamole/internal/model"
	"github.com/jraesly/reimagined-guacamole/internal/remote"
	"github.com/jraesly/reimagined-guacamole/internal/report"
	"github.com/jraesly/reimagined-guacamole/internal/scan"
	"github.com/jraesly/reimagined-guacamole/internal/suggest"
)

var version = "0.1.0-dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "fit":
		err = runFit(os.Args[2:])
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
  probe fit [flags] [target ...]                   fit models to this machine
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
		rep.Models = append(rep.Models, report.Build(m, nil, nil, budget.GB, opts, info.BandwidthGBs))
	}
	for _, r := range refs {
		addRemote(r)
	}
	for _, t := range targets {
		var m *model.Model
		var merr error
		if t.IsDir {
			m, merr = model.FromMLXDir(t.Path)
		} else {
			m, merr = model.FromGGUF(t.Path)
		}
		if merr != nil {
			rep.Models = append(rep.Models, report.ModelResult{Name: firstName(t), Path: t.Path, Error: merr.Error()})
			continue
		}
		if t.Owner != "" && (m.Owner == "" || t.Source == "ollama") {
			m.Owner = t.Owner
			m.Baseline, m.BaselineNote = model.Classify(m.Owner, m.Name)
		}
		if len(t.Names) > 0 {
			m.Name = t.Names[0]
		}
		rep.Models = append(rep.Models, report.Build(m, aliases(t), t.Extras, budget.GB, opts, info.BandwidthGBs))
	}
	if len(rep.Models) == 0 && !*doSuggest {
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
		if strings.HasSuffix(p, "k") {
			mult, p = 1024, strings.TrimSuffix(p, "k")
		}
		n, err := strconv.ParseUint(p, 10, 64)
		if err != nil || n == 0 {
			return nil, fmt.Errorf("bad context size %q (use e.g. 32k or 32768)", part)
		}
		out = append(out, n*mult)
	}
	return out, nil
}
