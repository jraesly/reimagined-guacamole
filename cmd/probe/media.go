package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/jraesly/reimagined-guacamole/internal/hw"
	"github.com/jraesly/reimagined-guacamole/internal/media"
	"github.com/jraesly/reimagined-guacamole/internal/scan"
)

// mediaTasks are the --for values served by the media path rather than the
// LLM path.
var mediaTasks = map[string]media.Kind{"image": media.Image, "video": media.Video, "tts": media.TTS, "speech": media.Speech}

// mediaRow is one model in the media report.
type mediaRow struct {
	Name       string            `json:"name"`
	Path       string            `json:"path"`
	Kind       media.Kind        `json:"kind"`
	Family     string            `json:"family"`
	Source     string            `json:"source"`
	DTypes     string            `json:"dtypes"`
	Components []media.Component `json:"components,omitempty"`
	Fit        media.Result      `json:"fit"`
	Note       string            `json:"note"`
	Warnings   []string          `json:"warnings,omitempty"`
	Error      string            `json:"error,omitempty"`
}

type mediaReport struct {
	Hardware hw.Info      `json:"hardware"`
	Budget   hw.Budget    `json:"budget"`
	Task     string       `json:"task"`
	Hosts    []media.Host `json:"hosts"`
	Models   []mediaRow   `json:"models"`
}

// runMediaFit reports image/video/TTS/speech models on this machine (or the
// explicit paths given) as weights-only fits plus family notes.
func runMediaFit(task string, paths []string, info hw.Info, budget hw.Budget, activationGB float64, asJSON bool, out io.Writer) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	want := mediaTasks[task]
	rep := mediaReport{Hardware: info, Budget: budget, Task: task, Hosts: media.DetectHosts(home)}
	var found []scan.MediaFound
	if len(paths) == 0 {
		found = scan.Media(home)
	} else {
		for _, p := range paths {
			k, ok := media.Detect(p)
			if !ok {
				rep.Models = append(rep.Models, mediaRow{Name: p, Path: p, Error: "not a recognised image/video/TTS/speech model (safetensors, diffusers dir, diffusion GGUF, whisper ggml, kokoro/xtts)"})
				continue
			}
			found = append(found, scan.MediaFound{Path: p, Name: p, Kind: string(k), Source: "path"})
		}
	}
	for _, f := range found {
		if want != "" && media.Kind(f.Kind) != want {
			continue
		}
		m, err := media.Load(f.Path)
		if err != nil {
			rep.Models = append(rep.Models, mediaRow{Name: f.Name, Path: f.Path, Kind: media.Kind(f.Kind), Source: f.Source, Error: err.Error()})
			continue
		}
		rep.Models = append(rep.Models, mediaRow{
			Name: f.Name, Path: f.Path, Kind: m.Kind, Family: m.Family, Source: f.Source, DTypes: m.DTypeSummary,
			Components: m.Components, Fit: media.Fit(m, budget.GB, media.Options{ActivationGB: activationGB}),
			Note: media.FamilyNote(m.Family), Warnings: m.Warnings,
		})
	}
	sort.Slice(rep.Models, func(i, j int) bool { return rep.Models[i].Name < rep.Models[j].Name })
	if asJSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	writeMediaText(out, rep, activationGB)
	return nil
}

func writeMediaText(w io.Writer, rep mediaReport, activationGB float64) {
	fmt.Fprintf(w, "Machine   %s, %.0f GB, budget %.1f GB\n", rep.Hardware.Chip, rep.Hardware.RAMGB, rep.Budget.GB)
	if len(rep.Hosts) > 0 {
		names := make([]string, 0, len(rep.Hosts))
		for _, h := range rep.Hosts {
			names = append(names, fmt.Sprintf("%s (%s)", h.Name, strings.Join(h.Kinds, "/")))
		}
		fmt.Fprintf(w, "Runtimes  %s\n", strings.Join(names, ", "))
	} else {
		fmt.Fprintln(w, "Runtimes  none detected (ComfyUI, Draw Things, MLX-audio, whisper.cpp)")
	}
	if len(rep.Models) == 0 {
		fmt.Fprintf(w, "\nNo %s models found on this machine. probe looks in ComfyUI/Draw Things model folders, the Hugging Face cache and whisper.cpp model dirs; pass a path to check a specific file.\n", rep.Task)
		return
	}
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "model\tkind\tfamily\tweights GB\tdtypes\tverdict")
	for _, m := range rep.Models {
		if m.Error != "" {
			fmt.Fprintf(tw, "%s\t%s\t-\t-\t-\terror: %s\n", m.Name, m.Kind, m.Error)
			continue
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.1f\t%s\t%s\n", m.Name, m.Kind, m.Family, m.Fit.WeightsGB, m.DTypes, m.Fit.Verdict)
	}
	tw.Flush()
	fmt.Fprintln(w)
	seen := map[string]bool{}
	for _, m := range rep.Models {
		if m.Error == "" && m.Note != "" && !seen[m.Family] {
			seen[m.Family] = true
			fmt.Fprintf(w, "  %s: %s\n", m.Family, m.Note)
		}
		for _, wn := range m.Warnings {
			fmt.Fprintf(w, "  warning (%s): %s\n", m.Name, wn)
		}
	}
	if activationGB > 0 {
		fmt.Fprintf(w, "\nVerdicts include %.1f GB of activations you supplied (inferred); weights are measured from the files.\n", activationGB)
	} else {
		fmt.Fprintln(w, "\nweights-only = the weights fit; activation memory depends on resolution/frames and is not in any file header. Pass --activation-gb from your own runs to get a full verdict.")
	}
}
