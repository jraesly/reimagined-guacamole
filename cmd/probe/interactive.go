package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/jraesly/reimagined-guacamole/internal/measure"
	"github.com/jraesly/reimagined-guacamole/internal/suggest"
)

// menu is the interactive entry point used when probe runs with no
// arguments on a terminal. Every choice maps onto a normal command line, so
// nothing here has behaviour of its own.
var menu = []struct {
	label string
	hint  string
}{
	{"What fits on this machine", "probe fit"},
	{"Find a model for a task", "probe fit --suggest --online --for <task>"},
	{"Measure my installed models", "probe fit --measure"},
	{"Watch what my coding harness sends", "probe capture --upstream <server>"},
	{"Image / video / TTS / speech models here", "probe fit --for image|video|tts|speech"},
	{"Quit", ""},
}

func runInteractive(in io.Reader, out io.Writer) error {
	rd := bufio.NewReader(in)
	for first := true; ; first = false {
		fmt.Fprintln(out, "\nprobe — what do you want to do?")
		for i, m := range menu {
			fmt.Fprintf(out, "  %d) %-38s %s\n", i+1, m.label, m.hint)
		}
		choice, err := ask(rd, out, "> ")
		if err != nil {
			if first {
				// Nothing to read at all (e.g. stdin is /dev/null): behave
				// like a non-interactive invocation.
				usage()
			}
			return nil
		}
		switch choice {
		case "1":
			if err := runFit(nil); err != nil {
				fmt.Fprintln(out, "probe:", err)
			}
		case "2":
			task, err := askTask(rd, out)
			if err != nil {
				return nil
			}
			args := []string{"--suggest", "--online"}
			if task != "" {
				args = append(args, "--for", task)
			}
			if err := runFit(args); err != nil {
				fmt.Fprintln(out, "probe:", err)
			}
		case "3":
			fmt.Fprintln(out, "This loads each installed model once (a minute or so each) and records its decode speed.")
			if ok, _ := askYesNo(rd, out, "Continue? [y/N] "); ok {
				if err := runFit([]string{"--measure"}); err != nil {
					fmt.Fprintln(out, "probe:", err)
				}
			}
		case "4":
			upstream, err := askUpstream(rd, out)
			if err != nil {
				return nil
			}
			if upstream == "" {
				continue
			}
			report := fmt.Sprintf("probe-capture-%s.md", time.Now().Format("20060102-150405"))
			fmt.Fprintf(out, "Starting the proxy; the Markdown report will be written to ./%s when you press Ctrl-C.\n", report)
			return runCapture([]string{"--upstream", upstream, "--report", report})
		case "5":
			kind, err := ask(rd, out, "which kind? [image/video/tts/speech, Enter for all] ")
			if err != nil {
				return nil
			}
			args := []string{}
			if kind = strings.ToLower(strings.TrimSpace(kind)); kind != "" {
				args = append(args, "--for", kind)
			} else {
				args = append(args, "--for", "image")
				fmt.Fprintln(out, "(showing image models; run probe fit --for video|tts|speech for the others)")
			}
			if err := runFit(args); err != nil {
				fmt.Fprintln(out, "probe:", err)
			}
		case "6", "q", "quit", "exit", "":
			return nil
		default:
			fmt.Fprintln(out, "pick 1-6")
		}
	}
}

func ask(rd *bufio.Reader, out io.Writer, prompt string) (string, error) {
	fmt.Fprint(out, prompt)
	line, err := rd.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

func askYesNo(rd *bufio.Reader, out io.Writer, prompt string) (bool, error) {
	a, err := ask(rd, out, prompt)
	if err != nil {
		return false, err
	}
	a = strings.ToLower(a)
	return a == "y" || a == "yes", nil
}

func askTask(rd *bufio.Reader, out io.Writer) (string, error) {
	names := make([]string, 0, len(suggest.Tasks))
	for t := range suggest.Tasks {
		names = append(names, t)
	}
	sort.Strings(names)
	fmt.Fprintln(out, "Which task? (Enter for all)")
	for i, n := range names {
		fmt.Fprintf(out, "  %d) %-10s %s\n", i+1, n, suggest.Tasks[n])
	}
	a, err := ask(rd, out, "> ")
	if err != nil {
		return "", err
	}
	for i, n := range names {
		if a == fmt.Sprint(i+1) || strings.EqualFold(a, n) {
			return n, nil
		}
	}
	if a == "" {
		return "", nil
	}
	if why, bad := suggest.Unsupported[strings.ToLower(a)]; bad {
		fmt.Fprintln(out, why)
		return "", nil
	}
	fmt.Fprintf(out, "unknown task %q; showing all\n", a)
	return "", nil
}

// askUpstream offers whichever local servers answer right now.
func askUpstream(rd *bufio.Reader, out io.Writer) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	type server struct{ name, url string }
	var found []server
	if measure.OllamaAvailable(ctx) {
		found = append(found, server{"Ollama", measure.OllamaBase + "/v1"})
	}
	if measure.LMStudioAvailable(ctx) {
		found = append(found, server{"LM Studio", measure.LMStudioBase + "/v1"})
	}
	fmt.Fprintln(out, "Which model server does your harness talk to?")
	for i, s := range found {
		fmt.Fprintf(out, "  %d) %-10s %s  (running)\n", i+1, s.name, s.url)
	}
	fmt.Fprintf(out, "  %d) another URL\n", len(found)+1)
	if len(found) == 0 {
		fmt.Fprintln(out, "  (neither Ollama nor LM Studio answered; start one, or enter its URL)")
	}
	a, err := ask(rd, out, "> ")
	if err != nil {
		return "", err
	}
	for i, s := range found {
		if a == fmt.Sprint(i+1) {
			return s.url, nil
		}
	}
	if a == fmt.Sprint(len(found)+1) {
		return ask(rd, out, "base URL (e.g. http://localhost:8080/v1): ")
	}
	if strings.HasPrefix(a, "http") {
		return a, nil
	}
	return "", nil
}

// isTerminal reports whether stdin is an interactive terminal.
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
