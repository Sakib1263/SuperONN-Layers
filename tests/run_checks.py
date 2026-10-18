"""Run a backend suite in a time/RSS-bounded subprocess.

Linux /proc samples the child and its descendants. Summed RSS includes shared
pages more than once; it is a conservative process-tree estimate, not GPU or
activation memory. RSS accounting is unavailable on other operating systems.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def tree_rss_mib(pid):
    pending, seen, kib = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = Path("/proc") / str(current)
        try:
            data = (directory / "status").read_text()
            fields = dict(line.split(":", 1) for line in data.splitlines() if ":" in line)
            kib += int(fields.get("VmRSS", "0 kB").split()[0])
            for task in (directory / "task").iterdir():
                try:
                    pending.extend(int(p) for p in (task / "children").read_text().split())
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    return kib / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("base", "tensorflow", "torch", "interop", "execution"))
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--rss-mib", type=float, default=3072)
    parser.add_argument("--test", action="append", help="Run only these unittest names")
    args = parser.parse_args()
    if not (0 < args.seconds < float("inf") and 0 < args.rss_mib < float("inf")):
        parser.error("time and RSS budgets must be finite and positive")
    command = [sys.executable, "-B", "-m", "unittest", "-v",
               *(args.test or ["tests.test_" + args.suite])]
    child = subprocess.Popen(command, start_new_session=os.name == "posix")
    start, peak, stopped = time.monotonic(), 0, None
    measured = sys.platform.startswith("linux") and Path("/proc").exists()
    try:
        while child.poll() is None:
            if measured:
                peak = max(peak, tree_rss_mib(child.pid))
            if time.monotonic() - start > args.seconds:
                stopped = "timeout"
            if peak > args.rss_mib:
                stopped = "RSS budget exceeded"
            if stopped:
                break
            time.sleep(.1)
    finally:
        if child.poll() is None:
            def stop(sig):
                try:
                    if os.name == "posix":
                        os.killpg(child.pid, sig)
                    elif sig == signal.SIGTERM:
                        child.terminate()
                    else:
                        child.kill()
                except ProcessLookupError:
                    pass
            stop(signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
                child.kill()
                child.wait()
    print(json.dumps({"suite": args.suite, "seconds": round(time.monotonic() - start, 3),
                      "peak_process_tree_rss_mib": round(peak, 1) if measured else None,
                      "rss_accounting_available": measured, "stopped": stopped,
                      "returncode": child.returncode}), flush=True)
    return 1 if stopped else child.returncode


if __name__ == "__main__":
    sys.exit(main())
