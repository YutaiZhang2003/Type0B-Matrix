"""Native double-Virasoro coefficients for physical continuum NSRR blocks.

The C++ ordinary and inserted pipelines cover equal and opposite HJS signs,
respectively. Physical continuum momenta p are passed as note momenta P=i*p.
There is no physical PBW fallback or saved-reference input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
METHOD = "C++ double-Virasoro CCY with ordinary/inserted fermion recovery"


def executable():
    path = Path(os.environ.get("TYPE0B_NSRR_BINARY", ROOT / "C++/bin/ramond")).resolve()
    if not path.is_file():
        raise RuntimeError("Native NSRR executable missing; run make -C C++ from the repository root")
    sources = list((ROOT / "C++/include/ramond").glob("*.hpp")) + list((ROOT / "C++/src").glob("*.cpp"))
    if any(source.stat().st_mtime_ns > path.stat().st_mtime_ns for source in sources):
        raise RuntimeError("Native NSRR executable is older than its sources; run make -C C++")
    return path


def implementation_hashes():
    paths = [Path(__file__), executable()]
    paths += sorted((ROOT / "C++/include/ramond").glob("*.hpp"))
    paths += sorted((ROOT / "C++/src").glob("*.cpp"))
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def physical_rows(record):
    rows = {}
    for row in record["coefficients"]:
        a, left, right, d = row["exponents"]
        if left != right or a < 0 or left < 0 or d < 0 or d % 2 or len(row["values"]) != 8:
            raise ValueError("Invalid native physical NSRR monomial")
        key = a, 2 * left, d
        if key in rows:
            raise ValueError("Duplicate native NSRR monomial")
        values = tuple(complex(float(v["real"]), float(v["imag"])) for v in row["values"])
        if any(not (math.isfinite(v.real) and math.isfinite(v.imag)) for v in values):
            raise ArithmeticError("Nonfinite native NSRR coefficient")
        rows[key] = values
    cutoff = 2 * record["total_q_level"]
    expected = {(a, b, c) for a in range(cutoff + 1)
                for b in range(0, cutoff - a + 1, 2)
                for c in range(0, cutoff - a - b + 1, 2)}
    if rows.keys() != expected:
        raise ArithmeticError("Incomplete native NSRR coefficient table")
    return rows


class NativeNSRR:
    def __init__(self, b, physical_momenta, cutoff, primary_parity=0, dps=40):
        self.binary = executable()
        self.b = repr(float(b))
        self.momenta = ["0," + repr(float(p)) for p in physical_momenta]
        self.cutoff = cutoff
        self.primary_parity = primary_parity
        self.dps = int(dps)
        if self.dps != 0 and self.dps < 30:
            raise ValueError("Native precision must be zero or at least 30 digits")
        self.records = {}
        self.components = {}

    def physical_components(self, form_parity, eta_left, eta_right):
        key = form_parity, eta_left, eta_right
        if key in self.components:
            return self.components[key]
        if form_parity not in (0, 1) or eta_left not in (-1, 1) or eta_right not in (-1, 1):
            raise ValueError("Invalid NSRR form parity or HJS signs")
        mode = "ordinary" if eta_left == eta_right else "inserted"
        with tempfile.TemporaryDirectory(prefix="type0b-nsrr-") as directory:
            output = Path(directory) / "block.json"
            command = [str(self.binary), "--mode", mode, "--level", str(self.cutoff),
                       "--dps", str(self.dps), "--b", self.b,
                       "--P1", self.momenta[0], "--P2", self.momenta[1], "--P3", self.momenta[2],
                       "--p", str(self.primary_parity), "--f", str(form_parity),
                       "--eta", str(eta_left), "--sector-policy", "error", "--json", str(output)]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=300)
            if completed.returncode:
                raise ArithmeticError(f"Native NSRR {key} failed at P={self.momenta}: {completed.stderr.strip()}")
            record = json.loads(output.read_text())
        expected = dict(status="computed", mode=mode, truncation="total", total_q_level=self.cutoff,
                        b=self.b, momenta=self.momenta, p=self.primary_parity, f=form_parity,
                        etas=[eta_left, eta_right], dps=self.dps, sector_policy="error",
                        branching_method="stored_recursion")
        for field, value in expected.items():
            if record.get(field) != value:
                raise ArithmeticError(f"Native NSRR metadata mismatch: {field}")
        self.components[key] = physical_rows(record)
        self.records[key] = {field: record[field] for field in
                             ("mode", "dps", "precision_bits", "diagnostics", "counts", "timing_seconds")}
        return self.components[key]

    @property
    def ward_residual_maximum(self):
        return max((record["diagnostics"]["maximum_ward_residual"] for record in self.records.values()), default=0.0)

    def diagnostics(self):
        return dict(method=METHOD, explicit_PBW_completion_calls=0,
                    physical_PBW_used=False, native_precision_digits=self.dps,
                    native_pipeline_calls=len(self.records),
                    branching_ward_residual=self.ward_residual_maximum,
                    native_channels=[dict(channel=list(key), **record) for key, record in self.records.items()])
