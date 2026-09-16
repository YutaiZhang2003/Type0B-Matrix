"""Pointwise NSRR blocks with independently converged global descendants.

Branching and residue orders are separate. No polynomial coefficient cache
or PBW fallback supplies the result. Slots are (NS,R_one,R_zero); momenta
are physical nonnegative Liouville momenta, continued to iP at the boundary.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
METHOD = "native double-Virasoro with independently resummed global descendants"


def executable():
    path = Path(os.environ.get("TYPE0B_NSRR_RESUMMED_BINARY", ROOT / "C++/bin/ramond_resummed")).resolve()
    sources = list((ROOT / "C++/include/ramond").glob("*.hpp")) + [ROOT / "C++/tools/resummed_main.cpp"]
    if not path.is_file() or any(p.stat().st_mtime_ns > path.stat().st_mtime_ns for p in sources):
        raise RuntimeError("Build the resummed evaluator with make -C C++ bin/ramond_resummed")
    return path


def encoded(value):
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("finite complex input required")
    return f"{value.real!r},{value.imag!r}"


class ResummedNSRR:
    def __init__(self, b, physical_momenta, *, branch_level, recursion_order,
                 branch_truncation="per-edge", global_tolerance=1e-13,
                 global_max_shell=64, primary_parity=0, dps=40,
                 output_directory=None, timeout=7200):
        self.binary = executable()
        self.binary_sha256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.b = float(b)
        self.momenta = tuple(map(float, physical_momenta))
        if not math.isfinite(self.b) or self.b <= 0 or self.b == 1:
            raise ValueError("b must be finite, positive and different from one")
        if len(self.momenta) != 3 or any(not math.isfinite(p) or p < 0 for p in self.momenta):
            raise ValueError("three nonnegative physical momenta are required in (NS,R1,R0) order")
        for value in (branch_level, recursion_order, global_max_shell, primary_parity, dps):
            if int(value) != value:
                raise ValueError("orders, parity and precision must be integers")
        if min(branch_level, recursion_order) < 0 or primary_parity not in (0, 1):
            raise ValueError("invalid cutoff or primary parity")
        if branch_truncation not in ("total", "per-edge"):
            raise ValueError("invalid branching domain")
        if not 0 < global_tolerance < 1 or not 4 <= global_max_shell <= 256:
            raise ValueError("invalid global accuracy controls")
        if dps != 0 and dps < 30:
            raise ValueError("precision must be zero or at least 30 digits")
        self.options = dict(branch_level=int(branch_level), recursion_order=int(recursion_order),
                            branch_truncation=branch_truncation, global_tolerance=float(global_tolerance),
                            global_max_shell=int(global_max_shell), primary_parity=int(primary_parity), dps=int(dps))
        self.directory = Path(output_directory) if output_directory else None
        self.timeout = timeout
        self.records = {}
        self.values = {}

    def physical_values(self, q_values, form_parity, eta_left, eta_right):
        q = tuple(map(complex, q_values))
        if len(q) != 3 or any(not math.isfinite(abs(z)) or abs(z) >= 1 for z in q):
            raise ValueError("three plumbing parameters with |q| < 1 required")
        if form_parity not in (0, 1) or eta_left not in (-1, 1) or eta_right not in (-1, 1):
            raise ValueError("invalid form parity or eta")
        key = (q, form_parity, eta_left, eta_right)
        if key in self.values:
            return self.values[key]
        mode = "ordinary" if eta_left == eta_right else "inserted"
        request = dict(b=repr(self.b), momenta=[encoded(1j*p) for p in self.momenta],
                       q_values=list(map(encoded,q)), mode=mode, f=form_parity,
                       etas=[eta_left,eta_right], **self.options)
        expected_provenance = dict(request=request, binary_sha256=self.binary_sha256)
        identity = hashlib.sha256(json.dumps(expected_provenance,sort_keys=True).encode()).hexdigest()[:20]
        with tempfile.TemporaryDirectory(prefix="type0b-nsrr-resummed-") as temporary:
            directory = self.directory or Path(temporary)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"block-{identity}.json"
            provenance = directory / f"block-{identity}.request.json"
            binary_hash = hashlib.sha256(self.binary.read_bytes()).hexdigest()
            if binary_hash != self.binary_sha256:
                raise ArithmeticError("resummed executable changed during evaluation")
            if path.exists():
                if not provenance.exists() or json.loads(provenance.read_text()) != expected_provenance:
                    raise ArithmeticError("resummed checkpoint input/binary hash mismatch")
            else:
                command = [str(self.binary), "--mode", mode, "--b", repr(self.b),
                           "--p", str(self.options["primary_parity"]), "--f", str(form_parity),
                           "--eta", str(eta_left), "--json", str(path)]
                for option in ("branch_level", "recursion_order", "branch_truncation",
                               "global_tolerance", "global_max_shell", "dps"):
                    command += ["--"+option.replace("_","-"), str(self.options[option])]
                for e in range(3):
                    command += [f"--P{e+1}", request["momenta"][e], f"--q{e+1}", request["q_values"][e]]
                with path.with_suffix(".log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout)
                if completed.returncode:
                    message = path.with_suffix(".log").read_text()[-4000:]
                    raise ArithmeticError(f"Resummed NSRR {key[1:]} failed: {message}")
                provenance.write_text(json.dumps(expected_provenance, indent=2)+"\n")
            record = json.loads(path.read_text())
        expected = dict(status="computed", method="double_virasoro_global_resummed", b=repr(self.b),
                        recursion_domain="physical total level per copy; punctured diagonal downward closure",
                        momenta=request["momenta"], q_values=request["q_values"], mode=mode,
                        p=self.options["primary_parity"], f=form_parity, etas=[eta_left,eta_right],
                        branch_level=self.options["branch_level"], recursion_order=self.options["recursion_order"],
                        branch_truncation=self.options["branch_truncation"], dps=self.options["dps"],
                        global_tolerance=self.options["global_tolerance"], global_maximum_shell=self.options["global_max_shell"])
        for field, value in expected.items():
            if record.get(field) != value:
                raise ArithmeticError(f"Resummed NSRR metadata mismatch: {field}")
        values = tuple(complex(float(v["real"]),float(v["imag"])) for v in record["physical_values"])
        if len(values) != 8 or any(not math.isfinite(abs(v)) for v in values):
            raise ArithmeticError("invalid resummed physical parity vector")
        self.records[key] = record
        self.values[key] = values
        return values

    def channels(self, q_values, *, use_parity_symmetry=True):
        """Return all eight form/sign channels, before ordinary lift projection."""
        if not use_parity_symmetry:
            return {(f,e,ep): self.physical_values(q_values,f,e,ep)
                    for f,e,ep in itertools.product((0,1),(1,-1),(1,-1))}
        if self.options["primary_parity"] != 0:
            raise ValueError("three-call reconstruction is validated for primary parity zero only")
        zero = {(1,1):self.physical_values(q_values,0,1,1),
                (-1,-1):self.physical_values(q_values,0,-1,-1),
                (1,-1):self.physical_values(q_values,0,1,-1)}
        zero[-1,1] = zero[1,-1]
        out = {}
        for e,ep in itertools.product((1,-1),repeat=2):
            out[0,e,ep] = zero[e,ep]
            out[1,e,ep] = tuple(-1j*(-1)**((j&1)+((j>>1)&1))*zero[e,ep][j^4] for j in range(8))
        return out

    @staticmethod
    def project(values, lifts):
        lifts = tuple(lifts)
        if len(lifts) != 3 or any(l not in (-1,1) for l in lifts):
            raise ValueError("three literal square-root lifts required")
        return sum(v*math.prod(lifts[e]**((j>>e)&1) for e in range(3)) for j,v in enumerate(values))

    def diagnostics(self):
        return dict(method=METHOD, physical_PBW_used=False, **self.options,
                    native_pipeline_calls=len(self.records),
                    channels=[dict(channel=list(key[1:]), **record["diagnostics"], seconds=record["seconds"])
                              for key,record in self.records.items()])
