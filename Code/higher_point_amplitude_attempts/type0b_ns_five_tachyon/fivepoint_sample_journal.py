"""Append-only persistent sample checkpoints for long integrations."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path


class SampleJournal:
    """Single-writer journal, synced every 16 new samples and on close.

    A hard kill can require recomputing at most the last 16 unsynced samples.
    A partial last record is discarded on reopening; damaged complete records
    and source/config mismatches fail closed. No samples are silently dropped.
    """

    def __init__(self, path, signature, sync_every=16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values, self.reused, self.pending = {}, 0, 0
        self.sync_every = int(sync_every)
        if self.sync_every < 1:
            raise ValueError("sync_every must be positive")
        self.handle = self.path.open('a+b')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.seek(0)
            header = self.handle.readline()
            expected = {'schema':'fivepoint-sample-journal-v1','signature':signature}
            if header:
                if not header.endswith(b'\n') or json.loads(header) != expected:
                    raise ValueError("checkpoint source/config signature mismatch")
            else:
                self.handle.write((json.dumps(expected)+'\n').encode())
                self.handle.flush()
                os.fsync(self.handle.fileno())
            last_good = self.handle.tell()
            for line in self.handle:
                if not line.endswith(b'\n'):
                    self.handle.seek(last_good)
                    self.handle.truncate()
                    break
                record = json.loads(line)
                key, pair = record['key'], record['value']
                value = complex(*pair)
                if not math.isfinite(value.real) or not math.isfinite(value.imag):
                    raise ValueError("nonfinite checkpoint sample")
                if key in self.values and self.values[key] != pair:
                    raise ValueError("conflicting checkpoint sample")
                self.values[key] = pair
                last_good = self.handle.tell()
            self.handle.seek(0, os.SEEK_END)
        except BaseException:
            self.handle.close()
            raise

    def evaluate(self, key, compute):
        if key in self.values:
            self.reused += 1
            return complex(*self.values[key])
        value = complex(compute())
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ArithmeticError("refusing to checkpoint a nonfinite sample")
        pair = [value.real,value.imag]
        self.handle.write((json.dumps({'key':key,'value':pair},allow_nan=False)+'\n').encode())
        self.values[key] = pair
        self.pending += 1
        if self.pending >= self.sync_every:
            self.flush()
        return value

    def flush(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.pending = 0

    def close(self):
        if not self.handle.closed:
            try:
                self.flush()
            finally:
                self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@contextmanager
def open_sample_checkpoint(path, signature, storage='snapshot'):
    if storage == 'journal':
        with SampleJournal(Path(path).with_suffix('.jsonl'),signature) as checkpoint:
            yield checkpoint
    elif storage == 'snapshot':
        from fivepoint_runtime import SampleCheckpoint
        yield SampleCheckpoint(path,signature)
    else:
        raise ValueError("unknown sample checkpoint storage")
