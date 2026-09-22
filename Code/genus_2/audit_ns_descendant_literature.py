#!/usr/bin/env python3
"""Descendant-level comparison with NS literature vertex conventions.

The independent vertex below uses the component-ordered commutators in
Hadasz--Jaskolski--Suchanek (hep-th/0611266), equation (4.11). It does not
call the Human-Note Ward or global-three-form implementation. Both use
the same algebra module, whose Gram matrices are separately compared with
the explicit literature matrices. A frozen pre-change snapshot tests the
NS descendant-basis switch in the numerical and exact sewing modules.
"""
from __future__ import annotations

import argparse
import cmath
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import sympy as s

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"Code/c_Recursion"), str(ROOT/"Code/genus_2")]
from ns_genus2_symbolic_low_order import (
    ExactNSVermaModule, ExactNSDescendantThreeForm,
    state_twice_level, state_parity,
)
from all_ns_reflected_sewing import LIFTS

OUTPUT = ROOT/"Data Set/ns_literature_basis_switch_20260915"


class ComponentNSVertex:
    """Ordered NS vertex, with both primary three-point seeds set to one.

    Mode integers are doubled, matching the algebra-module representation.
    The inserted component's parity is determined by the boundary states;
    it must be included when commuting a supercurrent through the vertex.
    """

    def __init__(self, c, weights):
        self.c = s.sympify(c)
        self.weights = tuple(map(s.sympify, weights))
        self.modules = tuple(ExactNSVermaModule(c=self.c, weight=h)
                             for h in self.weights)
        self._reverse = None

    def reverse(self):
        if self._reverse is None:
            self._reverse = ComponentNSVertex(self.c, self.weights[::-1])
            self._reverse._reverse = self
        return self._reverse

    def acted(self, states, slot, mode):
        result = s.S.Zero
        for word, coefficient in self.modules[slot].mode_action(mode, states[slot]):
            changed = list(states)
            changed[slot] = word
            result += coefficient*self.value(*changed)
        return result

    @lru_cache(maxsize=None)
    def boundary(self, bra, star, ket):
        """HJS (4.11), with (4.14)--(4.15) for an empty bra."""
        h3, h2, h1 = self.weights
        if not bra:
            if not ket:
                return s.S.One
            return (-1)**(star*state_parity(ket))*self.reverse().boundary(ket, star, ())
        (kind, twice_mode), tail = bra[0], bra[1:]
        mode = -s.Rational(twice_mode, 2)
        tail_level = s.Rational(state_twice_level(tail), 2)
        ket_level = s.Rational(state_twice_level(ket), 2)
        if kind == "L":
            factor = h3+tail_level-h1-ket_level+mode*(h2+s.Rational(star, 2))
            result = factor*self.boundary(tail, star, ket)
            crossing = 1
        else:
            factor = 1 if not star else h3+tail_level-h1-ket_level+2*mode*h2
            result = factor*self.boundary(tail, 1-star, ket)
            # This is the parity of the component V before moving G across it.
            crossing = (-1)**(state_parity(bra)+state_parity(ket))
        for word, coefficient in self.modules[2].mode_action((kind, -twice_mode), ket):
            result += crossing*coefficient*self.boundary(tail, star, word)
        return s.cancel(result)

    @lru_cache(maxsize=None)
    def value(self, bra, middle, ket):
        if middle in ((), (("G", -1),)):
            return self.boundary(bra, int(bool(middle)), ket)
        (kind, twice_mode), tail = middle[0], middle[1:]
        mode = -s.Rational(twice_mode, 2)
        states = (bra, tail, ket)
        result = s.S.Zero
        if kind == "L":
            n = int(mode)
            if n == 1:
                exponent = (self.weights[0]+s.Rational(state_twice_level(bra), 2)
                            -self.weights[1]-s.Rational(state_twice_level(tail), 2)
                            -self.weights[2]-s.Rational(state_twice_level(ket), 2))
                return s.cancel(exponent*self.value(*states))
            for m in range(max(0, state_twice_level(bra)//2-n)+1):
                result += s.binomial(n-2+m, m)*self.acted(states, 0, ("L", 2*(n+m)))
            for m in range(state_twice_level(ket)//2+2):
                result += (-1)**n*s.binomial(n-2+m, m)*self.acted(states, 2, ("L", 2*(m-1)))
        else:
            k = mode
            for m in range(max(0, int(s.floor(s.Rational(state_twice_level(bra), 2)-k)))+1):
                result += s.binomial(k-s.Rational(3, 2)+m, m)*self.acted(states, 0, ("G", int(2*(k+m))))
            # Deforming the G contour past a component vertex contributes its
            # own parity. This sign is derived from the supercommutator; it is
            # not inferred from the parity of the middle descendant alone.
            crossing = (-1)**(state_parity(bra)+state_parity(ket)+int(k-s.Rational(1, 2)))
            for m in range((state_twice_level(ket)+1)//2+1):
                result += crossing*s.binomial(k-s.Rational(3, 2)+m, m)*self.acted(states, 2, ("G", 2*m-1))
        return s.cancel(result)


def vertex_comparison():
    c = s.Rational(81, 5)
    weights = tuple(map(s.Rational, ("7/10", "11/10", "13/10")))
    literature = ComponentNSVertex(c, weights)
    human = ExactNSDescendantThreeForm(c=c, weights=weights)
    count = 0
    failures = []
    for levels in itertools.product(range(5), repeat=3):
        if sum(levels) > 6:
            continue
        for states in itertools.product(*(m.basis(n) for m, n in zip(literature.modules, levels))):
            a = sum(levels) % 2
            expected = (-1)**(a*state_parity(states[2]))*literature.value(*states)
            actual = human.value(*states)
            count += 1
            if s.cancel(actual-expected) != 0:
                failures.append(dict(levels=levels,states=states,human=str(actual),expected=str(expected)))
    assert not failures, failures
    return dict(count=count,failures=failures,
                dictionary="rho_HN^a(x1,x2,x3)=(-1)^(a*parity(x3))*rho_component(x1,x2,x3)",
                scope="Even NS highest states; all PBW triples with each edge level <=2 and total level <=3.")


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def source_hashes():
    sources = [ROOT/"Code/c_Recursion"/name for name in (
        "mixed_ns_ramond_descendant_blocks.py", "ns_genus2_symbolic_low_order.py",
        "ns_genus_c_recursion_checks.py", "ns_genus12_finite_c_check.py",
        "ns_genus2_glasses_finite_c_check.py", "ns_human_convention.py")]
    shared = ROOT/"Code/c_Recursion/ns_pbw_basis.py"
    if shared.exists():
        sources.append(shared)
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}


def snapshot():
    """Capture actual theta/glasses sewing before or after the basis change."""
    from ns_genus12_finite_c_check import DirectThetaOracle, level_tuples
    from ns_genus2_glasses_finite_c_check import DirectGlassesOracle
    from ns_genus2_symbolic_low_order import ExactDirectThetaOracle
    from mixed_ns_ramond_descendant_blocks import NSVermaModule
    from all_ns_reflected_sewing import reflected_blocks

    fixtures = ((s.Rational(81, 5), (s.Rational(7, 10),s.Rational(11, 10),s.Rational(13, 10))),
                (s.Rational(149, 4), (s.Rational(73, 100),s.Rational(91, 100),s.Rational(117, 100))))
    levels = tuple(level_tuples(6))
    coefficients, partitions = [], []
    q = (.012*cmath.exp(.16j), .021*cmath.exp(-.22j), .016*cmath.exp(.37j))
    for fixture, (c, weights) in enumerate(fixtures):
        theta = DirectThetaOracle(c=complex(c), weights=tuple(map(complex, weights)))
        glasses = DirectGlassesOracle(c=complex(c), weights=tuple(map(complex, weights)))
        exact = ExactDirectThetaOracle(c=c, weights=weights)
        sums = {channel:{a:{lift:0j for lift in LIFTS} for a in (0,1)}
                for channel in ("theta", "glasses")}
        for ns in levels:
            sector_theta, sector_glasses = sum(ns)%2, ns[2]%2
            values = {
                "theta":theta.coefficient(twice_levels=ns,sectors=(sector_theta,sector_theta)),
                "glasses":glasses.coefficient(ns,sector_glasses),
            }
            exact_value = str(exact.coefficient(ns)) if fixture == 0 else None
            for channel, value in values.items():
                sector = sector_theta if channel == "theta" else sector_glasses
                coefficients.append(dict(fixture=fixture,channel=channel,twice_levels=ns,
                    sector=sector,value=pair(value),
                    exact=exact_value if channel == "theta" else None))
                monomial = math.prod(cmath.exp(n/2*cmath.log(z)) for n,z in zip(ns,q))
                for lift in LIFTS:
                    # Theta coefficients use (infinity,one,zero); canonical
                    # lift triples use (zero,one,infinity).
                    signs = lift[::-1] if channel == "theta" else lift
                    character = math.prod(sign**(n%2) for sign,n in zip(signs,ns))
                    sums[channel][sector][lift] += value*monomial*character
        primary = cmath.exp(sum(complex(h)*cmath.log(z) for h,z in zip(weights,q)))
        constants = (1.2*.7, .4*.9)  # Independent, nonzero even/odd products.
        for channel in ("theta", "glasses"):
            converted = {a:reflected_blocks(sums[channel][a],a) for a in (0,1)} if channel == "theta" else None
            for lift in LIFTS:
                blocks = [sums[channel][a][lift] for a in (0,1)]
                terms = [constants[a]*abs(primary*blocks[a])**2 for a in (0,1)]
                reflected = None if converted is None else sum(
                    constants[a]*abs(primary*converted[a][lift])**2 for a in (0,1))
                partitions.append(dict(fixture=fixture,channel=channel,lift=lift,
                    q_slots=list(map(pair,q)),weights=list(map(str,weights)),
                    primary=pair(primary),even_descendant=pair(blocks[0]),odd_descendant=pair(blocks[1]),
                    even_term=terms[0],odd_term=terms[1],literal_partition=sum(terms),
                    reflected_partition=reflected))
    numeric = NSVermaModule(c=16.2,weight=.7)
    exact_module = ExactNSVermaModule(c=s.Rational(81,5),weight=s.Rational(7,10))
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes(),
        numeric_bases={n:numeric.basis(n) for n in range(15)},
        exact_bases={n:exact_module.basis(n) for n in range(15)},
        coefficients=coefficients,partitions=partitions,
        scope="Finite-level before/after equality in two generic fixtures, theta and glasses, both nonzero structure products, all four lifts. No new momentum integral or spin-transport claim.")


def literature_matrices():
    """Generic torus insertion: BG Appendix B in the printed PBW order.

    BG uses c_BG=2*c/3. T_N below is their inserted matrix minus the Gram
    matrix, so a zero external weight gives the identity trace exactly.
    """
    c, h, d = s.symbols("c h d")
    module = ExactNSVermaModule(c=c, weight=h)
    vertex = ComponentNSVertex(c, (h,d,h))
    gram = {
        1:s.Matrix([[2*h]]),
        2:s.Matrix([[2*h]]),
        3:s.Matrix([[2*h*(2*h+1),4*h],[4*h,2*h+2*c/3]]),
        4:s.Matrix([[4*h*(2*h+1),6*h,8*h],
                    [6*h,4*h+c/2,3*h+c],
                    [8*h,3*h+c,4*h*h-2*h+4*c*h/3+2*c]]),
    }
    insertion_minus_gram = {
        1:s.Matrix([[-d]]),
        2:s.Matrix([[d*(d-1)]]),
        3:s.Matrix([[-d*(d*d-2*d*h-d+4*h+1),-d*(d+1)],
                    [-d*(d+1),-3*d]]),
        4:s.Matrix([
            [d*(d-1)*(d*d-d+8*h+2),2*d*(d-1)*(d+1),d*(d-1)*(3*d+2)],
            [2*d*(d-1)*(d+1),4*d*(d-1),6*d*(d-1)],
            [d*(d-1)*(3*d+2),6*d*(d-1),d*(11*d-8*h-9-2*c/3)],
        ]),
    }
    rows = []
    for n in range(1,5):
        actual_gram = module.gram_matrix(n)
        basis = module.basis(n)
        inserted = s.Matrix([[vertex.value(bra,(),ket) for ket in basis] for bra in basis])
        assert (actual_gram-gram[n]).applyfunc(s.cancel) == s.zeros(len(basis))
        assert (inserted-actual_gram-insertion_minus_gram[n]).applyfunc(s.cancel) == s.zeros(len(basis))
        coefficient = s.cancel(s.trace(actual_gram.inv()*inserted))
        identity_coefficient = s.cancel(coefficient.subs(d,0))
        assert identity_coefficient == len(basis)
        rows.append(dict(twice_level=n,basis=basis,gram=str(actual_gram),
                         inserted_minus_gram=str(insertion_minus_gram[n]),
                         torus_coefficient=str(coefficient),identity_coefficient=int(identity_coefficient)))
    return dict(gram_entries=15,inserted_entries=15,identity_coefficients=[1,1,1,2,3],
                central_charge_dictionary="c_BG=2*c/3",rows=rows)


def compare_snapshots(before, after):
    """Compare complex coefficients first, then norms and phases separately."""
    def as_complex(value):
        return complex(*value)

    basis_permutations = {}
    for n in range(15):
        old = before["numeric_bases"][str(n)]
        new = after["numeric_bases"][n]
        # JSON represents tuples as lists; normalize without changing order.
        new = json.loads(json.dumps(new))
        assert before["exact_bases"][str(n)] == old
        assert json.loads(json.dumps(after["exact_bases"][n])) == new
        assert sorted(map(str,old)) == sorted(map(str,new))
        basis_permutations[n] = [old.index(state) for state in new]

    def coefficient_key(row):
        return row["fixture"], row["channel"], tuple(row["twice_levels"])

    old_coefficients = {coefficient_key(row):row for row in before["coefficients"]}
    assert set(old_coefficients) == {coefficient_key(row) for row in after["coefficients"]}
    coefficient_rows, exact_count = [], 0
    for row in after["coefficients"]:
        old = old_coefficients[coefficient_key(row)]
        a, b = as_complex(old["value"]), as_complex(row["value"])
        error = abs(a-b)/max(1,abs(a),abs(b))
        assert error < 1e-11, (row,error)
        if row["exact"] is not None:
            assert s.cancel(s.sympify(row["exact"])-s.sympify(old["exact"])) == 0
            exact_count += 1
        coefficient_rows.append(dict(fixture=row["fixture"],channel=row["channel"],
            twice_levels=str(tuple(row["twice_levels"])),sector=row["sector"],
            before_real=a.real,before_imag=a.imag,after_real=b.real,after_imag=b.imag,
            scaled_error=error,exact_match=row["exact"] is not None))

    def partition_key(row):
        return row["fixture"], row["channel"], tuple(row["lift"])

    old_partitions = {partition_key(row):row for row in before["partitions"]}
    assert set(old_partitions) == {partition_key(row) for row in after["partitions"]}
    partition_rows = []
    for row in after["partitions"]:
        old = old_partitions[partition_key(row)]
        assert row["primary"] == old["primary"]
        result = dict(fixture=row["fixture"],channel=row["channel"],lift=str(tuple(row["lift"])))
        for sector in ("even","odd"):
            a, b = as_complex(old[sector+"_descendant"]), as_complex(row[sector+"_descendant"])
            relative_complex = abs(a-b)/max(abs(a),abs(b),1e-300)
            relative_norm = abs(abs(a)-abs(b))/max(abs(a),abs(b),1e-300)
            phase_error = abs(cmath.phase(b/a)) if a and b else None
            assert relative_complex < 1e-11
            assert phase_error is None or phase_error < 1e-11
            result.update({sector+"_relative_complex_error":relative_complex,
                           sector+"_relative_norm_error":relative_norm,
                           sector+"_phase_error_radians":phase_error})
        for name in ("even_term","odd_term","literal_partition","reflected_partition"):
            if row[name] is None:
                assert old[name] is None
                continue
            relative = abs(row[name]-old[name])/max(abs(row[name]),abs(old[name]),1e-300)
            assert relative < 1e-11, (name,row,relative)
            result[name+"_relative_error"] = relative
        partition_rows.append(result)
    summary = dict(
        exact_theta_coefficients=exact_count,
        numerical_sewing_coefficients=len(coefficient_rows),
        genus_two_evaluations=len(partition_rows),
        max_scaled_coefficient_error=max(row["scaled_error"] for row in coefficient_rows),
        max_relative_block_error=max(row[sector+"_relative_complex_error"] for row in partition_rows for sector in ("even","odd")),
        max_relative_block_norm_error=max(row[sector+"_relative_norm_error"] for row in partition_rows for sector in ("even","odd")),
        max_block_phase_error_radians=max(row[sector+"_phase_error_radians"] or 0 for row in partition_rows for sector in ("even","odd")),
        max_relative_partition_error=max(value for row in partition_rows for key,value in row.items() if "partition_relative_error" in key),
        primary_factors_identical=True,
        before_to_after_vector_permutations=basis_permutations,
        cutoff="Total physical descendant level <= 3 in both genus-two channels.",
        scope=after["scope"],
    )
    return summary, coefficient_rows, partition_rows


def write_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        # Glasses rows do not use the theta-specific reflected contraction.
        for row in rows:
            for key in row:
                if key not in writer.fieldnames:
                    writer.fieldnames.append(key)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-before",action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.capture_before:
        path = args.output/"before.json"
        if path.exists():
            raise FileExistsError(f"Preserve the existing pre-change snapshot: {path}")
        result = snapshot()
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(dict(output=str(path),coefficients=len(result["coefficients"]),
                              partitions=len(result["partitions"]))))
    else:
        before = json.loads((args.output/"before.json").read_text())
        after = snapshot()
        result, coefficient_rows, partition_rows = compare_snapshots(before,after)
        result["component_vertex_dictionary"] = vertex_comparison()
        result["literature_torus_matrices"] = literature_matrices()
        for name, data in (("after.json",after),("summary.json",result)):
            (args.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
        write_csv(args.output/"coefficient_comparison.csv",coefficient_rows)
        write_csv(args.output/"partition_comparison.csv",partition_rows)
        print(json.dumps({key:value for key,value in result.items()
                          if key not in ("before_to_after_vector_permutations","literature_torus_matrices")},indent=2))


if __name__ == "__main__":
    main()
