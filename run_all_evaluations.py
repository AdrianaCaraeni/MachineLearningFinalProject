"""Run evaluation scripts sequentially; append stdout/stderr to RERUN_RESULTS.txt with flushing."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "RERUN_RESULTS.txt")

SCRIPTS = [
    "evaluate_digits.py",
    "evaluate_parkinsons.py",
    "evaluate_rice.py",
    "evaluate_credit_approval.py",
    "evaluate_gnb_q4.py",
    "evaluate_ensemble.py",
]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(LOG, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n\n=== run_all_evaluations.py {datetime.now().isoformat()} ===\n")

        for script in SCRIPTS:
            path = os.path.join(ROOT, script)
            log.write(f"\n\n######## {script} ########\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, "-u", path],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            proc.wait()
            log.write(f"\n[exit {proc.returncode}]\n")
            log.flush()
            if proc.returncode != 0:
                print(f"FAILED: {script} exit {proc.returncode}", file=sys.stderr)
                sys.exit(proc.returncode)

        # Optional EC2 — skip if data missing
        titanic = os.path.join(ROOT, "titanic.csv")
        if os.path.isfile(titanic):
            log.write("\n\n######## evaluate_titanic.py ########\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(ROOT, "evaluate_titanic.py")],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            proc.wait()
            log.flush()
        else:
            log.write("\n\n[skipped evaluate_titanic.py — titanic.csv not found]\n")
            log.flush()

    print(f"Wrote {LOG}")


if __name__ == "__main__":
    main()
