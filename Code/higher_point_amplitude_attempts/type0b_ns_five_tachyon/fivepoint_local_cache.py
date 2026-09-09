"""Stage a private coefficient database on node-local storage, with copy-back.

Only coefficient storage moves. Sample checkpoints and final results remain
on persistent storage. SQLite backup copies committed data even if an
exception leaves the evaluator's connection open.
"""
from contextlib import closing, contextmanager
import fcntl
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import sys
import tempfile
import threading
import time


def _snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst)
                if dst.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise ValueError("coefficient snapshot failed SQLite integrity check")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def staged_coefficient_cache(persistent: Path, scratch_root: Path):
    """Yield (local_path, transfer_record), then atomically persist the database.

    The persistent .lock is the same lock used by CoefficientStore, preventing
    another direct or staged writer from changing this cache concurrently.
    SIGTERM becomes SystemExit so Slurm's grace period can run copy-back.
    SIGKILL/node loss can discard newly built tables; completed scalar sample
    checkpoints are independent of this regenerable cache.
    """
    persistent = Path(persistent)
    scratch_root = Path(scratch_root)
    persistent.parent.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    record = {"persistent_path": str(persistent), "copy_in_seconds": 0.0,
              "copy_out_seconds": 0.0, "copied_back": False}
    with persistent.with_suffix(persistent.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("persistent coefficient cache already owned") from error
        directory = tempfile.mkdtemp(prefix=persistent.stem + "-", dir=scratch_root)
        local = Path(directory) / persistent.name
        try:
            record["local_path"] = str(local)
            if persistent.exists():
                before = time.perf_counter()
                _snapshot(persistent, local)
                record["copy_in_seconds"] = time.perf_counter() - before
            previous = None
            if threading.current_thread() is threading.main_thread():
                previous = signal.getsignal(signal.SIGTERM)
                def terminate(signum, frame):
                    raise SystemExit(128 + signum)
                signal.signal(signal.SIGTERM, terminate)
            try:
                yield local, record
            finally:
                # A second termination signal must not interrupt copy-back.
                if previous is not None:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    if local.exists():
                        before = time.perf_counter()
                        _snapshot(local, persistent)
                        record["copy_out_seconds"] = time.perf_counter() - before
                        record["copied_back"] = True
                finally:
                    if previous is not None:
                        signal.signal(signal.SIGTERM, previous)
        finally:
            if record["copied_back"] or not local.exists():
                shutil.rmtree(directory)
            else:
                print(f"Coefficient copy-back incomplete; local recovery file: {local}", file=sys.stderr)
