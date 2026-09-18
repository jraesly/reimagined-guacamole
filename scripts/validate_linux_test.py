"""Offline integration checks: python3 scripts/validate_linux_test.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("validate-linux.sh")
MOCK = r'''
import json, os, pathlib, sys
command = pathlib.Path(sys.argv[0]).name
scenario = os.environ.get("SCENARIO", "pass")
if command == "probe":
    rows = [{"Context": c, "Verdict": v} for c, v in
            [(32768, "no"), (8192, "yes"), (65536, "no"), (16384, "tight")]]
    print(json.dumps({"hardware": {"Chip": "test GPU"}, "models": [
        {"name": "local.gguf", "aliases": ["test:latest"], "rows": rows},
        {"name": "no-alias.gguf", "rows": rows}]}))
elif command == "curl":
    body = json.loads(sys.argv[sys.argv.index("--data") + 1])
    with open("calls.jsonl", "a") as f:
        f.write(json.dumps(body) + "\n")
    if body.get("keep_alive") == 0:
        if scenario == "unload": sys.exit(22)
        print('{"done":true}')
        sys.exit(0)
    ctx = body["options"]["num_ctx"]
    pathlib.Path("context").write_text(str(ctx))
    if scenario == "transport": sys.exit(7)
    error = scenario == "error" and ctx == 32768
    response = {"error":"out of memory"} if error else {"done":True,"eval_count":8,"eval_duration":400000000}
    text = "not json" if scenario == "invalid" else json.dumps(response)
    pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_text(text)
    print("500" if error else "200", end="")
elif command == "ollama":
    ctx = int(pathlib.Path("context").read_text())
    if scenario == "ps_error": sys.exit(1)
    processor = "100% GPU" if ctx == 16384 else "48%/52% CPU/GPU"
    if scenario == "pessimistic": processor = "100% GPU"
    if scenario == "optimistic": processor = "100% CPU"
    fmt = "{:<24}{:<16}{:<12}{:<20}{:<12}{}"
    print(fmt.format("NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"))
    name = "unrelated:latest" if scenario == "missing" else "test:latest"
    print(fmt.format(name, "abc123", "8 GB", processor, ctx, "1 minute"))
elif command == "nvidia-smi":
    print("test GPU, 123.45")
'''


class ValidateLinuxTests(unittest.TestCase):
    def run_validator(self, scenario, dry=False):
        with tempfile.TemporaryDirectory(prefix=".validate-test-", dir=SCRIPT.parent) as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for command in ("probe", "curl", "ollama", "nvidia-smi"):
                path = bin_dir / command
                path.write_text("#!" + sys.executable + "\n" + MOCK)
                path.chmod(0o755)
            env = dict(os.environ, PROBE=str(bin_dir / "probe"),
                       PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], SCENARIO=scenario)
            run = subprocess.run(["bash", str(SCRIPT)] + (["--dry-run"] if dry else []),
                                 cwd=root, env=env, capture_output=True, text=True, timeout=30)
            reports = list(root.glob("docs/validation/*.json"))
            report = json.loads(reports[0].read_text()) if reports else None
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()] if (root / "calls.jsonl").exists() else []
            return run, report, calls

    def test_verdict_matrix_and_report(self):
        for scenario, expected in [("pass", ["PASS", "PASS"]), ("error", ["PASS", "PASS"]),
                                   ("pessimistic", ["PASS", "FAIL"]), ("optimistic", ["FAIL", "PASS"]),
                                   ("transport", ["FAIL", "FAIL"]), ("invalid", ["FAIL", "FAIL"]),
                                   ("ps_error", ["FAIL", "FAIL"]), ("missing", ["FAIL", "FAIL"]),
                                   ("unload", ["FAIL"])]:
            with self.subTest(scenario=scenario):
                run, report, calls = self.run_validator(scenario)
                self.assertEqual(run.returncode, int("FAIL" in expected), run.stdout + run.stderr)
                self.assertIsNotNone(report, run.stderr)
                results = report["results"]
                self.assertEqual([r["result"] for r in results], expected)
                self.assertEqual([r["ctx"] for r in results], [16384, 32768][:len(expected)])
                self.assertEqual(report["hardware"], {"Chip": "test GPU"})
                self.assertEqual(report["gpu_tool"], "nvidia-smi")
                self.assertTrue(report["uname"])
                if scenario == "pass":
                    self.assertEqual([r["tok_s"] for r in results], [20, 20])
                if scenario == "pessimistic":
                    self.assertIn("too pessimistic", run.stdout)
                if scenario != "unload":
                    self.assertEqual([c.get("keep_alive") for c in calls], ["1m", 0, "1m", 0])

    def test_dry_run_selects_boundaries_without_ollama(self):
        run, report, calls = self.run_validator("pass", dry=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIsNone(report)
        self.assertEqual(calls, [])
        self.assertIn('"num_ctx":16384', run.stdout)
        self.assertIn('"num_ctx":32768', run.stdout)
        self.assertNotIn('"num_ctx":8192', run.stdout)
        self.assertNotIn('"num_ctx":65536', run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
