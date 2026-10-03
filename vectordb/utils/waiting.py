"""Wait for the functions of a map without waiting forever.

Lithops' job monitor marks a function running when its start status arrives,
and ends a running function that outlasts the execution timeout with a
TimeoutError. A function that dies before it reports a start (for example a
handler that fails in the runtime) never reaches that watch, and the wait for
it goes on forever. ``collect`` watches the starts: once every function has
started, ``get_result`` takes over; until then, the wait gives up when no
function starts or finishes for the wait window (by default, the function
timeout plus a minute).
"""

import time

from vectordb.errors import BlocksDBError

WAIT_MARGIN_SEC = 60
POLL_SEC = 0.1


class FunctionsTimedOut(BlocksDBError, TimeoutError):
    """Some functions never started, and none started or finished within the
    wait window."""


def inactivity_window(fexec, margin_sec=WAIT_MARGIN_SEC):
    """Seconds without any function starting or finishing after which
    ``collect`` gives up on the functions that have not started: the
    backend's function timeout plus a margin, or None when the configuration
    declares no timeout at all."""
    backend = fexec.config["lithops"]["backend"]
    settings = fexec.config.get(backend) or {}
    limit = settings.get("runtime_timeout") or fexec.config["lithops"].get("execution_timeout")
    return int(limit) + margin_sec if limit else None


def collect(fexec, futures, window=None, poll_sec=POLL_SEC, clock=time.monotonic, sleep=time.sleep):
    """Return the results of ``futures``, as ``fexec.get_result`` does.

    ``window`` is the number of seconds without any function starting or
    finishing after which the wait for functions that have not started ends
    with ``FunctionsTimedOut``: None derives it from the backend, 0 waits
    forever as ``get_result`` does. Until every function has started, the
    wait reads, every ``poll_sec`` seconds, the states that the job monitor
    sets on the futures: no requests, no log lines and no signals, so it
    works from any thread. Then ``get_result`` logs, shows its progress bar,
    raises a function's error or Lithops' timeout, and returns the results:
    a function's error surfaces only once every function has started.
    """
    if window is None:
        window = inactivity_window(fexec)
    if not window:
        return fexec.get_result(futures)

    progress, last_progress = 0, clock()
    while True:
        stages = [_stage(future) for future in futures]
        if all(stages):
            return fexec.get_result(futures)
        if sum(stages) > progress:
            progress, last_progress = sum(stages), clock()
        elif clock() - last_progress > window:
            raise FunctionsTimedOut(
                f"{stages.count(0)} of {len(futures)} functions never started, and no function"
                f" started or finished in the last {window} s."
                " A function that dies before it reports a start is not seen by Lithops'"
                " timeout; look for it in the backend's logs."
            )
        sleep(poll_sec)


def _stage(future):
    """0 before the function starts, 1 while it runs, 2 once it has finished."""
    if future.ready or future.success or future.done:
        return 2
    return 1 if future.running else 0
