package media

import (
	"github.com/jraesly/reimagined-guacamole/internal/fit"
	"github.com/jraesly/reimagined-guacamole/internal/model"
	"math"
)

type Options struct {
	ActivationGB float64
	Note         string
}
type Result struct {
	WeightsGB, ActivationGB, TotalGB             float64
	Verdict, Basis                               string
	WeightsSource, ActivationSource, TotalSource model.Source
}

// Fit uses GiB, matching internal/fit. Supplied activations are inferred inputs:
// accepting a user's value does not prove it was measured on this machine.
func Fit(m *Model, budgetGB float64, opts Options) Result {
	r := Result{Verdict: "no", WeightsSource: model.Measured, ActivationSource: model.Inferred, TotalSource: model.Inferred}
	if m == nil || math.IsNaN(budgetGB) || math.IsInf(budgetGB, 0) || budgetGB < 0 || math.IsNaN(opts.ActivationGB) || math.IsInf(opts.ActivationGB, 0) || opts.ActivationGB < 0 {
		r.Basis = "invalid model, budget or activation input"
		return r
	}
	r.WeightsGB = float64(m.WeightsBytes) / fit.GiB
	r.TotalGB = r.WeightsGB
	if opts.ActivationGB > 0 {
		r.ActivationGB = opts.ActivationGB
		r.TotalGB += r.ActivationGB
		r.Verdict = string(fit.Classify(r.TotalGB, budgetGB))
		r.Basis = "measured stored weights plus user-supplied activations"
	} else if r.WeightsGB <= budgetGB {
		r.Verdict = "weights-only"
		r.Basis = "weights fit; activations scale with resolution/frames and are not modeled — pass --activation-gb from your own runs"
	} else {
		r.Basis = "measured weights alone exceed the budget; activations are not modeled"
	}
	if opts.Note != "" {
		r.Basis += "; " + opts.Note
	}
	return r
}

func FamilyNote(family string) string {
	switch family {
	case "sd15", "sd1":
		return "Stable Diffusion activations grow with image resolution and batch size."
	case "sdxl":
		return "SDXL activations grow with image resolution and batch size."
	case "sd3":
		return "Stable Diffusion transformer activations grow with image resolution and batch size."
	case "flux":
		return "Flux activations grow with image resolution and batch size."
	case "wan":
		return "Wan activations grow with resolution multiplied by frame count."
	case "hunyuan-video":
		return "HunyuanVideo activations grow with resolution multiplied by frame count."
	case "ltx":
		return "LTX activations grow with resolution multiplied by frame count."
	case "kokoro", "xtts", "csm", "bark", "sesame", "whisper":
		return "Speech activations are usually small relative to weights and these models often run on CPU, though audio length and batching still matter."
	default:
		return "Activation memory depends on the runtime and workload and is not proved by weight headers."
	}
}
