"""Supervisor that keeps main.py alive: runs it as a subprocess and
restarts it on non-zero exits, with exponential backoff.

- Child stdout/stderr go to logs/eden.log (rotated aside at daemon start if
  over 10MB).
- Non-zero exit -> restart after a backoff delay (2s, doubling to 30s cap);
  the backoff resets to 2s once the child has stayed up more than 60s.
- Exit code 0 -> clean shutdown, no restart.
- Exit code 2 (configuration problem, from main.py's failed critical
  startup checks) -> no restart; print "fix .env and run again".
- Ctrl+C/SIGTERM on the daemon terminates the child and exits without
  restarting.
"""
import os
import signal
import subprocess
import sys
import time

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "eden.log")
MAX_LOG_BYTES = 10 * 1024 * 1024

START_BACKOFF = 2.0
MAX_BACKOFF = 30.0
STAYED_UP_RESET_SECONDS = 60.0

# main.py exits with this code when its critical startup checks fail (bad or
# missing GEMINI_API_KEY etc.). Treat it as a permanent configuration problem,
# not a crash: restarting would just fail again in a loop.
EXIT_CONFIG_PROBLEM = 2


def _sigterm_handler(signum, frame):
    """Termination request surfaces as KeyboardInterrupt so the supervisor's
    existing clean-shutdown path (stop the child, don't restart) handles
    SIGTERM identically to Ctrl+C."""
    raise KeyboardInterrupt


def rotate_if_large(log_path: str = LOG_PATH) -> None:
    if not os.path.isfile(log_path):
        return
    if os.path.getsize(log_path) <= MAX_LOG_BYTES:
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    aside = f"{log_path}.{ts}"
    os.replace(log_path, aside)
    print(f"[daemon] log over {MAX_LOG_BYTES} bytes; rotated {log_path} -> {aside}")


def supervise(cmd: list[str], log, cmd_factory=None) -> None:
    """Supervise a child command until it exits cleanly or is interrupted.

    ``cmd_factory`` (used by tests to vary behavior per spawn) returns the
    command list for each launch; when omitted, ``cmd`` is used as-is.
    """
    start_backoff = START_BACKOFF
    max_backoff = MAX_BACKOFF
    stayed_up_reset = STAYED_UP_RESET_SECONDS
    if cmd_factory is None:
        cmd_factory = lambda: list(cmd)

    delay = start_backoff
    child = None
    try:
        while True:
            start = time.monotonic()
            cmd_list = cmd_factory()
            child = subprocess.Popen(cmd_list, stdout=log, stderr=subprocess.STDOUT)
            print(f"[daemon] child started (pid {child.pid}): {' '.join(cmd_list)}")
            rc = child.wait()
            uptime = time.monotonic() - start
            print(
                f"[daemon] child exited (pid {child.pid}, code {rc}, "
                f"uptime {uptime:.1f}s)",
                file=log,
                flush=True,
            )
            if uptime >= stayed_up_reset:
                delay = start_backoff
            if rc == 0:
                print("[daemon] child exited cleanly; stopping supervisor")
                break
            if rc == EXIT_CONFIG_PROBLEM:
                message = (
                    "[daemon] configuration problem, not restarting - "
                    "fix .env and run again"
                )
                print(message)
                print(message, file=log, flush=True)
                break
            print(f"[daemon] child crashed (code {rc}); restarting in {delay:.0f}s")
            print(
                f"[daemon] restarting in {delay:.0f}s",
                file=log,
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_backoff)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            print("[daemon] interrupt received; stopping child...")
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
        print("[daemon] daemon stopped, no restart")


def main() -> None:
    signal.signal(signal.SIGTERM, _sigterm_handler)
    os.makedirs(LOG_DIR, exist_ok=True)
    rotate_if_large()
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        print(
            f"[daemon] starting Eden at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            file=log,
            flush=True,
        )
        supervise([sys.executable, "-u", "main.py"], log)


if __name__ == "__main__":
    main()