"""Bounded-memory execution helpers; no physics or subtraction conventions.

The c-recursion scratch graph lives only while compiling missing coefficients.
Final coefficients remain at the original mpmath precision, optionally backed
by a single-writer, per-shard SQLite store (not a shared WAL on cluster NFS).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from functools import lru_cache
import fcntl
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import mpmath

from ns_multipoint_c_recursion import NSSphereLinearCRecursion, _validate_twice_levels


class BoundedLRU(MutableMapping):
    """Mapping-compatible LRU; eviction changes reuse, never numerical values."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self._data = OrderedDict()
        self.capacity = int(capacity)
        self.evictions = 0
        self.peak_entries = 0

    def __getitem__(self, key):
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key):
        return key in self._data

    def __delitem__(self, key):
        del self._data[key]

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __setitem__(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self) > self.capacity:
            self._data.popitem(last=False)
            self.evictions += 1
        self.peak_entries = max(self.peak_entries, len(self))


@lru_cache(maxsize=1)
def runtime_source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    # Conservative invalidation: physics, sampling, and serialization changes
    # invalidate both coefficient tables and sample checkpoints.
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += sorted((root / "c_Recursion").glob("*.py"))
    files += sorted((root / "bosonic_c1_one_to_n_reference" /
                     "reference_implementation" / "plumbing").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def coefficient_source_fingerprint() -> str:
    """Coefficient identity excludes integration, plotting and scheduling code.

    The complete c-recursion dependency directory remains conservatively
    hashed. Change the schema marker when changing final-table compilation or
    serialization. Sample checkpoints still use the complete runtime hash.
    """
    digest = hashlib.sha256(b"compact-c-final-tables-v2-json-dps12")
    directory = Path(__file__).resolve().parents[2] / "c_Recursion"
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CoefficientStore:
    """Durable final tables, with a bounded SQLite page cache and one writer."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = self.path.with_suffix(self.path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.close()
            raise RuntimeError("coefficient store already owned by another worker") from error
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS coefficients "
            "(key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()
        self.legacy_source = None
        if self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migration'"
        ).fetchone():
            metadata = dict(self.connection.execute("SELECT key,value FROM migration"))
            if metadata.get("coefficient_source") != coefficient_source_fingerprint():
                self.close()
                raise ValueError("migrated coefficient source mismatch")
            self.legacy_source = metadata["legacy_source"]
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key):
        row = self.connection.execute(
            "SELECT payload FROM coefficients WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key, payload):
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO coefficients(key,payload) VALUES (?,?)",
                (key, json.dumps(payload, separators=(",", ":"), allow_nan=False)),
            )
        self.writes += 1

    def close(self):
        self.connection.close()
        self._lock_handle.close()


class CompactCBlock(NSSphereLinearCRecursion):
    """Compile once, release recursion scratch, evaluate the same polynomial."""

    def __init__(self, *, coefficient_store=None, **kwargs):
        super().__init__(**kwargs)
        self.coefficient_store = coefficient_store
        self.final_coefficients = {}
        self.compiled_coefficients = 0
        self.peak_scratch_entries = 0
        self._store_key = None

    def _encode_number(self, value):
        return [mpmath.nstr(value.real, self.working_precision + 12),
                mpmath.nstr(value.imag, self.working_precision + 12)]

    def _load_final_coefficients(self):
        if self.coefficient_store is None or self._store_key is not None:
            return
        identity = {
            "version": 2,
            "source": coefficient_source_fingerprint(),
            "precision": self.working_precision,
            "pole_tolerance": self.pole_tolerance,
            "central_charge": self._encode_number(self.central_charge),
            "external_weights": [self._encode_number(x) for x in self.external_weights],
            "internal_weights": [self._encode_number(x) for x in self.internal_weights],
            "external_descendants": self.external_descendants,
            "vertex_sectors": self.vertex_sectors,
        }
        self._store_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = self.coefficient_store.get(self._store_key)
        if payload is None and self.coefficient_store.legacy_source is not None:
            legacy_identity = {**identity, "version": 1,
                               "source": self.coefficient_store.legacy_source}
            legacy_key = hashlib.sha256(json.dumps(
                legacy_identity, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            payload = self.coefficient_store.get(legacy_key)
        if payload is not None:
            self.final_coefficients = {
                tuple(levels): mpmath.mpc(real, imag)
                for levels, real, imag in payload
            }
            if not all(mpmath.isfinite(v) for v in self.final_coefficients.values()):
                raise ValueError("nonfinite coefficient in disk cache")

    def _prepare(self, levels):
        self._load_final_coefficients()
        missing = [level for level in levels if level not in self.final_coefficients]
        if not missing:
            return
        try:
            for level in missing:
                value = self._coefficient(
                    level, self.central_charge, self.internal_weights,
                    self.vertex_sectors,
                )
                if not mpmath.isfinite(value):
                    raise ArithmeticError("nonfinite compiled c coefficient")
                self.final_coefficients[level] = value
                self.compiled_coefficients += 1
        finally:
            self.peak_scratch_entries = max(
                self.peak_scratch_entries, len(self._coefficient_cache)
            )
            self.clear_cache()
        if self.coefficient_store is not None:
            self.coefficient_store.put(self._store_key, [
                [list(level), *self._encode_number(value)]
                for level, value in sorted(self.final_coefficients.items())
            ])

    def series_value(self, q_values, max_twice_levels, *,
                     max_total_twice_level=None, q_log_values=None,
                     minimum_twice_levels=None):
        maxima = _validate_twice_levels(max_twice_levels, expected_length=self.edge_count)
        minima = ((0,) * self.edge_count if minimum_twice_levels is None else
                  _validate_twice_levels(minimum_twice_levels, expected_length=self.edge_count))
        if len(q_values) != self.edge_count:
            raise ValueError("q_values must contain one entry per edge")
        if max_total_twice_level is not None and (
            isinstance(max_total_twice_level, bool)
            or not isinstance(max_total_twice_level, int)
            or max_total_twice_level < 0
        ):
            raise ValueError("max_total_twice_level must be non-negative or None")
        with mpmath.workdps(self.working_precision):
            q_tuple = tuple(mpmath.mpc(q) for q in q_values)
            if not all(mpmath.isfinite(q) for q in q_tuple):
                raise ValueError("q_values must be finite")
            logs = (tuple(mpmath.log(q) for q in q_tuple) if q_log_values is None
                    else tuple(mpmath.mpc(q) for q in q_log_values))
            if len(logs) != self.edge_count or not all(mpmath.isfinite(q) for q in logs):
                raise ValueError("q_log_values must contain one finite logarithm per edge")
            parities = self.compatible_level_parities()
            ranges = tuple(tuple(range(p, m + 1, 2)) for p, m in zip(parities, maxima))
            levels = [level for level in product(*ranges)
                      if all(a >= b for a, b in zip(level, minima))
                      and (max_total_twice_level is None or sum(level) <= max_total_twice_level)]
            self._prepare(levels)
            selected = {level: self.final_coefficients[level] for level in levels}
            # Factoring only the parity powers preserves the supplied NS log
            # lift, including conjugate lifts on the negative real axis.
            bases = tuple(mpmath.exp(log) for log in logs)

            def horner(axis, prefix):
                if axis == self.edge_count:
                    return selected.get(prefix, 0)
                value = mpmath.mpc(0)
                for level in reversed(ranges[axis]):
                    value = value * bases[axis] + horner(axis + 1, prefix + (level,))
                return value

            phase = mpmath.exp(sum(log * parity / 2 for log, parity in zip(logs, parities)))
            return phase * horner(0, ())


class SampleCheckpoint:
    """Atomic, source/config-validated journal of completed scalar samples."""

    def __init__(self, path: Path | str, signature: str):
        self.path = Path(path)
        self.signature = signature
        self.values = {}
        self.reused = 0
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            if payload.get("schema") != "fivepoint-samples-v1" or payload.get("signature") != signature:
                raise ValueError("checkpoint source/config signature mismatch")
            for key, pair in payload["values"].items():
                value = complex(*pair)
                if not math.isfinite(value.real) or not math.isfinite(value.imag):
                    raise ValueError("nonfinite checkpoint sample")
                self.values[key] = pair

    def evaluate(self, key, compute):
        if key in self.values:
            self.reused += 1
            return complex(*self.values[key])
        value = complex(compute())
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ArithmeticError("refusing to checkpoint a nonfinite sample")
        self.values[key] = [value.real, value.imag]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"schema": "fivepoint-samples-v1", "signature": self.signature,
                           "values": self.values}, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return value
