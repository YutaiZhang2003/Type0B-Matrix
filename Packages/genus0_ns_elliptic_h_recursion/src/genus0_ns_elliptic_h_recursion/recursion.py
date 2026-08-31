"""Public, parity-resolved coefficient tables and portable JSON persistence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import mpmath as mp

from .engine import RecursionEngine, KacPoleError
from .numbers import (number, nonnegative_integer, precision, parity_tuple,
                      twice_degree, central_charge_from_b, b_from_c)


NORMALIZATION = "human-note:NS:F=Lambda^(c)*prod(varrho_i^(h_i-c/24))*C_NS*H;unit-seed-H"


def _compositions(total, dimension):
    if dimension == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for tail in _compositions(total-first, dimension-1):
                yield (first,)+tail


def total_degree_indices(edge_count, order):
    """Yield twice-level tuples with sum <= 2*order, including all parities."""
    nonnegative_integer(edge_count, "edge_count")
    nonnegative_integer(order, "order")
    if not edge_count:
        raise ValueError("edge_count must be positive")
    for total in range(2*order+1):
        yield from _compositions(total, edge_count)


def _nome_logs(segment_nomes, edge_count, log_nomes):
    if len(segment_nomes) != edge_count:
        raise ValueError(f"expected {edge_count} segment nomes")
    nomes = tuple(number(p,"segment nome") for p in segment_nomes)
    if any(not p or abs(p) >= 1 for p in nomes):
        raise ValueError("nomes must obey 0<|p_i|<1; evaluate collision limits separately")
    if log_nomes is None:
        if any(mp.im(p) != 0 or mp.re(p) <= 0 for p in nomes):
            raise ValueError("complex/negative nomes require explicit coherent log_nomes")
        return tuple(mp.log(p) for p in nomes)
    if len(log_nomes) != edge_count:
        raise ValueError("log_nomes must have one entry per edge")
    logs = tuple(number(v,"log nome") for v in log_nomes)
    tol = mp.power(10,-(mp.mp.dps-8))
    if any(abs(mp.exp(l)-p) > tol*abs(p) for l,p in zip(logs,nomes)):
        raise ValueError("exp(log_nomes[i]) must match segment_nomes[i] at the table precision")
    return logs


@dataclass(frozen=True)
class RecursionTable:
    """All parity sectors, with a required sector choice for physical evaluation.

    The default sector is all-even, never an implicit sum over sectors.
    Coefficient keys are twice the powers of raw segment nomes p_i.
    """

    coefficients: Mapping
    order: int
    b: object
    central_charge: object
    external_weights: tuple
    internal_weights: tuple
    dps: int
    minimum_pole: object = None

    @property
    def edge_count(self):
        return len(self.internal_weights)

    @property
    def point_count(self):
        return len(self.external_weights)

    def evaluate(self, segment_nomes, *, parity=None, order=None, log_nomes=None):
        """Evaluate H in one sector; order can be integer or half-integer.

        Positive real p_i use positive square roots. Explicit logarithms are
        required for complex/negative p_i; changing log(p_i) by 2*pi*i
        multiplies an odd sector on that edge by -1.
        """
        return self._evaluate(segment_nomes,parity,order,log_nomes,False)

    def shell(self, segment_nomes, degree, *, parity=None, log_nomes=None):
        """One homogeneous shell, not an error bound on the omitted tail."""
        return self._evaluate(segment_nomes,parity,degree,log_nomes,True)

    def _evaluate(self, segment_nomes, parity, order, log_nomes, shell):
        with mp.workdps(self.dps):
            epsilon = parity_tuple(parity,self.edge_count)
            bound = twice_degree(self.order if order is None else order,self.order)
            logs = _nome_logs(segment_nomes,self.edge_count,log_nomes)
            terms = []
            for key,value in self.coefficients.items():
                if tuple(n%2 for n in key) != epsilon:
                    continue
                if sum(key) > bound or (shell and sum(key) != bound):
                    continue
                terms.append(value*mp.exp(mp.fsum(n*l/2 for n,l in zip(key,logs))))
            return +mp.fsum(terms)

    def evaluate_sectors(self, segment_nomes, *, order=None, log_nomes=None):
        """Return each sector separately; three-point constants are not included."""
        return {eps:self.evaluate(segment_nomes,parity=eps,order=order,log_nomes=log_nomes)
                for eps in product((0,1),repeat=self.edge_count)}

    def save(self, path):
        """Save an inspectable JSON table; no pickle or external workspace paths."""
        with mp.workdps(self.dps):
            def encode(value):
                return [mp.nstr(mp.re(value),self.dps+5),mp.nstr(mp.im(value),self.dps+5)]
            data = {"schema":1,"normalization":NORMALIZATION,"order":self.order,"dps":self.dps,
                    "b":encode(self.b),"central_charge":encode(self.central_charge),
                    "external_weights":list(map(encode,self.external_weights)),
                    "internal_weights":list(map(encode,self.internal_weights)),
                    "coefficients":[[list(k),encode(v)] for k,v in self.coefficients.items()]}
            if self.minimum_pole is not None:
                data["minimum_pole"] = dict(self.minimum_pole)
                data["minimum_pole"]["denominator"] = encode(self.minimum_pole["denominator"])
            Path(path).write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def compute_h_recursion(*, external_weights, internal_weights, order,
                        b=None, central_charge=None, dps=80,
                        pole_tolerance=None, progress=None):
    """Compute all parity sectors through total degree order at fixed weights.

    Supply exactly one of b and central_charge. The latter is the ordinary
    charge c=3/2+3*(b+1/b)^2. Generic nonconfluent parameters are required.
    The optional progress callback receives (completed_count,total_count).
    """
    precision(dps)
    nonnegative_integer(order,"order")
    if len(external_weights) < 4 or len(internal_weights) != len(external_weights)-3:
        raise ValueError("n>=4 external weights require n-3 internal weights")
    if (b is None) == (central_charge is None):
        raise ValueError("supply exactly one of b and central_charge")
    with mp.workdps(dps):
        b_value = b_from_c(central_charge) if b is None else number(b,"b")
        c_value = central_charge_from_b(b_value)
        external = tuple(number(x,"external weight") for x in external_weights)
        internal = tuple(number(x,"internal weight") for x in internal_weights)
        tolerance = mp.power(10,-(dps//2)) if pole_tolerance is None else number(pole_tolerance,"pole_tolerance")
        if mp.im(tolerance) or tolerance <= 0:
            raise ValueError("pole_tolerance must be positive and real")
        engine = RecursionEngine(b_value,external,internal,tolerance)
        keys = list(total_degree_indices(len(internal),order))
        coefficients = {}
        try:
            for done,key in enumerate(keys,1):
                value = engine.coefficient(key)
                if not mp.isfinite(value):
                    raise ArithmeticError(f"nonfinite coefficient at {key}")
                coefficients[key] = +value
                if progress is not None:
                    progress(done,len(keys))
            minimum = None if engine.minimum_pole is None else MappingProxyType(dict(engine.minimum_pole))
        finally:
            engine.clear()
        return RecursionTable(MappingProxyType(coefficients),order,+b_value,+c_value,
                              external,internal,dps,minimum)


def load_table(path):
    """Read and validate a JSON table created by RecursionTable.save."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("normalization") != NORMALIZATION:
        raise ValueError("unsupported table schema or normalization")
    dps,order = precision(data["dps"]),nonnegative_integer(data["order"],"order")
    with mp.workdps(dps):
        def decode(value):
            if not isinstance(value,list) or len(value) != 2:
                raise ValueError("invalid complex-number encoding")
            return number(mp.mpc(*value))
        b,c = decode(data["b"]),decode(data["central_charge"])
        external = tuple(map(decode,data["external_weights"]))
        internal = tuple(map(decode,data["internal_weights"]))
        if len(external) < 4 or len(internal) != len(external)-3:
            raise ValueError("invalid point or edge count")
        if abs(central_charge_from_b(b)-c) > mp.power(10,-(dps-8))*max(1,abs(c)):
            raise ValueError("central charge and b disagree")
        coefficients = {}
        for key,value in data["coefficients"]:
            if (not isinstance(key,list) or len(key) != len(internal)
                    or any(type(n) is not int or n < 0 for n in key)):
                raise ValueError("invalid twice-level index")
            key = tuple(key)
            if key in coefficients:
                raise ValueError("duplicate coefficient")
            coefficients[key] = decode(value)
        if set(coefficients) != set(total_degree_indices(len(internal),order)):
            raise ValueError("table has missing or out-of-cutoff coefficients")
        if coefficients[(0,)*len(internal)] != 1:
            raise ValueError("H must have unit seed")
        minimum = data.get("minimum_pole")
        if minimum is not None:
            minimum["denominator"] = decode(minimum["denominator"])
            minimum = MappingProxyType(minimum)
        return RecursionTable(MappingProxyType(coefficients),order,b,c,external,internal,dps,minimum)
