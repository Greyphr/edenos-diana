"""Supervisor that keeps main.py alive: runs it as a subprocess and
restarts it on non-zero exits, with exponential backoff.

- Child stdout+stderr are piped and forwarded line by line (background
  thread) to a logging.handlers.RotatingFileHandler on logs/eden.log,
  which rolls over to eden.log.1/.2/... on every write past MAX_LOG_BYTES
  (so a long-running child gets rotated logs even if it never exits), and
  mirrored to the daemon's own stdout to stay visible live.
- Non-zero exit -> restart after a backoff delay (2s, doubling to 30s cap);
  the backoff resets to 2s once the child has stayed up more than 60s.
- Exit code 0 -> clean shutdown, no restart.
- Exit code 2 (configuration problem, from main.py's failed critical
  startup checks) -> no restart; print "fix .env and run again".
- Ctrl+C/SIGTERM on the daemon terminates the child and exits without
  restarting.
"""
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "eden.log")
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

START_BACKOFF = 2.0
MAX_BACKOFF = 30.0
STAYED_UP_RESET_SECONDS = 60.0

# main.py exits with this code when its critical startup checks fail (bad or
# missing GEMINI_API_KEY etc.). Treat it as a permanent configuration problem,
# not a crash: restarting would just fail again in a loop.
EXIT_CONFIG_PROBLEM = 2

_LOGGER_SEED = 0


def _sigterm_handler(signum, frame):
    """Termination request surfaces as KeyboardInterrupt so the supervisor's
    existing clean-shutdown path (stop the child, don't restart) handles
    SIGTERM identically to Ctrl+C."""
    raise KeyboardInterrupt


def _make_logger(log_path: str = LOG_PATH) -> logging.Logger:
    """Build a logger whose file handler rolls the log over while running.

    A fresh logger per caller keeps the handler lifecycle simple (a long-
    lived daemon constructs one at startup; tests make short-lived ones for
    temp files) and avoids duplicate handlers on a shared logger.
    """
    global _LOGGER_SEED
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER_SEED += 1
    logger = logging.getLogger(f"eden.daemon.{_LOGGER_SEED}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _say(logger: logging.Logger, message: str) -> None:
    """Print to the daemon's terminal AND write through the rotating log."""
    logger.info(message)
    print(message, flush=True)


def _pump_output(child: subprocess.Popen, logger: logging.Logger) -> None:
    """Forward the child's stdout/stderr line by line to the rotating log
    and the daemon's own stdout, for the child's whole lifetime.

    Ends when the child exits and its pipe closes; daemon=True so an
    abandoned thread can never block shutdown.
    """
    try:
        stdout = child.stdout
        if stdout is None:
            return
        for line in stdout:
            _say(logger, line.rstrip("\n"))
    except Exception:
        pass


def supervise(cmd: list[str], logger: logging.Logger, cmd_factory=None) -> None:
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
    pump = None
    try:
        while True:
            start = time.monotonic()
            cmd_list = cmd_factory()
            child = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            print(f"[daemon] child started (pid {child.pid}): {' '.join(cmd_list)}")
            pump = threading.Thread(
                target=_pump_output, args=(child, logger), daemon=True
            )
            pump.start()
            rc = child.wait()
            if pump is not None:
                pump.join(timeout=5)
            uptime = time.monotonic() - start
            logger.info(
                "[daemon] child exited (pid %d, code %d, uptime %.1fs)",
                child.pid, rc, uptime,
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
                _say(logger, message)
                break
            print(f"[daemon] child crashed (code {rc}); restarting in {delay:.0f}s")
            _say(logger, f"[daemon] restarting in {delay:.0f}s")
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
        if pump is not None and pump.is_alive():
            pump.join(timeout=5)
        print("[daemon] daemon stopped, no restart")


def main() -> None:
    signal.signal(signal.SIGTERM, _sigterm_handler)
    logger = _make_logger()
    logger.info(
        "[daemon] starting Eden at %s", time.strftime("%Y-%m-%d %H:%M:%S")
    )
    supervise([sys.executable, "-u", "main.py"], logger)


if __name__ == "__main__":
    main()