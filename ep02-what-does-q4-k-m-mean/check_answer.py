#!/usr/bin/env python3
"""Run what a model wrote for prompt.txt against the three merge_intervals checks.

  python3 check_answer.py logs/prompt-UD-Q4_K_M.txt [more files...]

Takes the answer from a `llama cli -o` file: the text after "Assistant:", and after "[End thinking]" if that's
there. Strips any code fences, then runs it with each check in a temporary folder, using `timeout 5 python3 -I`.
An empty answer, or one that fails to run, counts as a failure (0 of 3). Prints one line per file and exits 0.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKS = [
    "assert [list(x) for x in merge_intervals([[1,3],[2,6],[8,10],[15,18]])] == [[1,6],[8,10],[15,18]]",
    "assert [list(x) for x in merge_intervals([[5,7],[1,2],[2,3]])] == [[1,3],[5,7]]",
    "assert list(merge_intervals([])) == []",
]


def answer_of(text: str) -> str:
    if "Assistant:" not in text:
        return ""
    a = text.split("Assistant:", 1)[1]
    if "[End thinking]" in a:
        a = a.split("[End thinking]", 1)[1]
    elif "[Start thinking]" in a:          # thinking never ended: no answer
        return ""
    a = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*$", "", a, flags=re.M)   # code fences
    return a.strip()


def run(code: str) -> dict:
    if not code:
        return {"ran": False, "passed": 0, "error": "empty answer (still thinking when -n ran out?)"}
    passed, error = 0, None
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "answer.py").write_text(code + "\n")
        for i, check in enumerate(CHECKS, 1):
            (Path(d) / f"check{i}.py").write_text(
                "exec(open('answer.py').read())\n" + check + "\n")
            p = subprocess.run(["timeout", "5", sys.executable, "-I", f"check{i}.py"], cwd=d,
                               capture_output=True, text=True)
            if p.returncode == 0:
                passed += 1
            elif error is None:
                last = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
                error = f"check {i}: " + ("timed out" if p.returncode == 124 else last)
    return {"ran": True, "passed": passed, "error": error}


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    results = {}
    for p in paths:
        text = Path(p).read_text(errors="replace")
        r = run(answer_of(text))
        r["verdict"] = "PASS" if r["passed"] == len(CHECKS) else "FAIL"
        results[p] = r
        print(f"{Path(p).name:<36} {r['verdict']}  {r['passed']}/{len(CHECKS)} checks"
              + (f"  ({r['error']})" if r["error"] else ""))
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main([a for a in sys.argv[1:] if not a.startswith("--")]))
