"""Write and print the Acorn 2.0 release-gate report.

Run: python release_gate_report.py

It reads saved evidence, judges the gate, writes data/release_gate.json for the
dashboards to fetch, and prints the verdict. It never trains, scores or trades,
and it exits 0 either way: a held release is a result, not a crash.
Failing the gate is normal. Promotion needs more chronological data, not a rerun.
"""
import json
import os
from pathlib import Path

import release_evidence
import release_gate

ROOT = Path(__file__).resolve().parent
REPORT_FILE = ROOT / "data" / "release_gate.json"


def report(evidence=None):
    """Collect the saved evidence (unless one is supplied) and judge the gate."""
    return release_gate.evaluate(evidence if evidence is not None else release_evidence.collect())


def write_report(payload, path=REPORT_FILE):
    """Atomically replace the gate report the dashboards read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    return path


def main():
    payload = report()
    write_report(payload)
    summary = payload["summary"]
    print(f"Release decision: keep {payload['release_name']} — "
          f"{summary['passed']}/{summary['total']} requirements pass")
    for area in (release_gate.AREA_DATA, release_gate.AREA_QUALITY,
                 release_gate.AREA_PAPER, release_gate.AREA_RELEASE):
        checks = [check for check in payload["checks"] if check["area"] == area]
        if not checks:
            continue
        print(f"\n{area}")
        for check in checks:
            print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['requirement']}")
            print(f"         {check['detail']}")
    if summary["failing"]:
        print("\nBlocking requirements: " + ", ".join(summary["failing"]))
    print(f"\nWrote {REPORT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
