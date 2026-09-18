package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/capture"
)

func runCapture(args []string) error {
	fs := flag.NewFlagSet("capture", flag.ContinueOnError)
	upstream := fs.String("upstream", "", "model server base URL, e.g. http://localhost:1234/v1 (LM Studio), http://localhost:8080 (llama-server) or http://localhost:11434 (Ollama)")
	listen := fs.String("listen", "127.0.0.1:8787", "address the proxy listens on")
	ndjson := fs.String("json", "", "write every turn as NDJSON to this file on exit")
	reportPath := fs.String("report", "", "write the Markdown session report to this file on exit")
	dump := fs.String("dump", "", "directory for redacted raw request/response bodies (off by default)")
	maxBodyMiB := fs.Int64("max-body", 32, "largest request body accounted, in MiB; bigger ones are forwarded unaccounted")
	if extra, err := parseInterleaved(fs, args); err != nil {
		return err
	} else if len(extra) > 0 {
		return fmt.Errorf("capture takes no positional arguments, got %q", extra)
	}
	if *upstream == "" {
		return errors.New("capture needs --upstream, e.g. --upstream http://localhost:1234/v1")
	}
	u, err := url.Parse(*upstream)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("--upstream must be an http(s) URL with a host, got %q", *upstream)
	}
	if _, _, err := net.SplitHostPort(*listen); err != nil {
		return fmt.Errorf("--listen must be host:port, got %q", *listen)
	}

	p, err := capture.New(capture.Options{
		Upstream: u,
		MaxBody:  *maxBodyMiB << 20,
		DumpDir:  *dump,
		OnTurn:   func(t capture.Turn) { fmt.Println(capture.LiveLine(t)) },
	})
	if err != nil {
		return err
	}
	srv := &http.Server{Addr: *listen, Handler: p, ReadHeaderTimeout: 30 * time.Second}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w", *listen, err)
	}
	started := time.Now()
	fmt.Printf("probe capture %s\n", version)
	fmt.Printf("  listening  http://%s/v1  →  %s\n", *listen, u)
	fmt.Printf("  point your harness's OpenAI-compatible base URL at http://%s/v1 and use it normally\n", *listen)
	fmt.Printf("  request contents are not stored%s; press Ctrl-C for the session report\n\n", dumpNote(*dump))

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ln) }()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	select {
	case err := <-errCh:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-sig:
		fmt.Println()
	}
	// Give in-flight streams a moment to finish, then cut them so their
	// accounting completes before the report is built.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		_ = srv.Close()
	}
	if !p.Wait(3 * time.Second) {
		fmt.Fprintln(os.Stderr, "probe: some in-flight turns did not finish accounting; report may be incomplete")
	}

	turns := p.Turns()
	env := capture.Env{Version: version, OS: runtime.GOOS, Arch: runtime.GOARCH, Upstream: u.String(), Listen: *listen, Started: started, Ended: time.Now()}
	capture.WriteSummary(os.Stdout, turns, env)
	if *ndjson != "" {
		if err := writeFile(*ndjson, func(f *os.File) error { return capture.WriteNDJSON(f, turns) }); err != nil {
			return err
		}
		fmt.Printf("\nturns written to %s\n", *ndjson)
	}
	if *reportPath != "" {
		if err := writeFile(*reportPath, func(f *os.File) error { capture.WriteMarkdown(f, turns, env); return nil }); err != nil {
			return err
		}
		fmt.Printf("report written to %s\n", *reportPath)
	}
	return nil
}

func dumpNote(dir string) string {
	if dir == "" {
		return ""
	}
	return " except redacted copies in " + dir
}

func writeFile(path string, fn func(*os.File) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := fn(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
