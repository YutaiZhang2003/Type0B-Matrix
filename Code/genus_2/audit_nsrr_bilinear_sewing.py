#!/usr/bin/env python3
"""Derive and audit the physical NSRR matrix, including descendant BPZ duals."""
from __future__ import annotations

import argparse
import cmath
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ("Code", "Code/genus_2", "Code/c_Recursion", "Code/double_virasoro/nsrr"):
    sys.path.insert(0, str(ROOT/directory))
import numpy as np
import sympy as sp
from nsrr_bilinear_sewing import (CHANNELS, BASE_PROJECTION_LIFTS, projected_components,
    nsrr_bilinear_matrix, contract_nsrr_bilinear)
from nsrr_bilinear_state_oracle import PhysicalNSRRBPZ, ERJ, ELJ, DJ, U8, U8BAR
from nsrr_genus2_block import HumanNSRRThetaOracle
from ramond_pbw_generalized_ward import RamondPBWModule, word_parity, word_level
from recombine_saved_genus2_coefficient_ledger import Inputs, encode, decode, csum, object_digest, relative, write_csv
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS

DATA = ROOT/"Data Set"
OUTPUT = DATA/"nsrr_bilinear_sewing_20260915"


def clifford_derivation():
    """Exact 2-family trace, without fitting a matrix to descendants."""
    assert sp.simplify(ELJ.T*DJ*ERJ) == sp.eye(2)
    ks, rows = {}, []
    for a, b, eta in product((0, 1), (0, 1), (1, -1)):
        rh = sp.diag(1, sp.I*eta) if a == 0 else sp.Matrix([[0, 1], [sp.I*eta, 0]])
        ra = sp.diag(1, -sp.I*eta) if b == 0 else sp.Matrix([[0, 1], [-sp.I*eta, 0]])
        k = sp.zeros(2)
        for e, f, al, be, ga, de in product((0, 1), repeat=6):
            # Both trinion inputs are KETS. EL belongs in the Gram dual.
            k[e, f] += (ERJ[2*al+be, e]*ERJ[2*ga+de, f]
                       * (-1)**(b*(a+al+ga)+be*ga)*rh[al, ga]*ra[be, de])
        ks[a, b, eta] = sp.simplify(k)
    expected = {(0, 0): lambda t: sp.diag(1, t),
                (0, 1): lambda t: sp.Matrix([[0, U8BAR], [t*U8, 0]]),
                (1, 0): lambda t: sp.Matrix([[0, U8], [t*U8BAR, 0]]),
                (1, 1): lambda t: sp.diag(-sp.I, sp.I*t)}
    for (a, b, t), k in ks.items():
        assert sp.simplify(k-expected[a, b](t)) == sp.zeros(2)
    # Products of (F0,F1) and (Ft0,Ft1), relative to F0 Ft0.
    amplitudes = sp.Matrix([[1, sp.I*(-1)**b, -sp.I*(-1)**a, (-1)**(a+b)]
                            for a, b in product((0, 1), repeat=2)])
    for s, r, eta, etap in product((1, -1), repeat=4):
        table = []
        for a, b in product((0, 1), repeat=2):
            value = sum((-1)**((a+b)*(e+f)+e*f+a*b)*s**(a+b)*r**e
                        *ks[a, b, eta][e, f]*ks[a, b, etap][e, f]
                        for e, f in product((0, 1), repeat=2))
            table.append(sp.simplify(value))
        reduced = sp.simplify(amplitudes.inv()*sp.Matrix(table)/2).reshape(2, 2)
        wanted = sp.Matrix([[1, s], [s, 1]])/2 if eta*etap == -r else sp.zeros(2)
        assert reduced == wanted
        rows.append(dict(s=s, r=r, eta=eta, etap=etap, family_sum=list(map(str, table)),
                         matrix_per_cLcR=str(reduced)))
    return dict(ket_embedding=str(ERJ), dual_embedding=str(ELJ),
                graded_clifford_metric=str(DJ), exact_family_reductions=rows)


def identity_gram_checks():
    """Compare J-adapted BPZ Grams with independent identity two-point Ward data.

    Moving the R insertion at 1 to infinity uses f(z)=z/(1-z):
    B(l,r)=(-1)^(N_l+Nt_l) T_1(exp(L1+Lt1)l, exp(-L1-Lt1)r).
    This constructs the BPZ matrix independently of the Clifford formula.
    """
    oracle = PhysicalNSRRBPZ(h=0, beta3=sp.I*sp.Rational(2, 5))
    mod = oracle.modules[0]
    beta = oracle.anti_betas[0]
    # In the PHYSICAL two-family basis anti G0 is represented by the
    # same-i module at -beta, with the family rephasing below. This differs
    # from the ambient HJS anti basis used inside ramond_basis_gram.
    amod = RamondPBWModule(oracle.anti_c/24-beta**2, -beta, oracle.anti_c)
    @lru_cache(None)
    def action(st, anti):
        hw, aw, e = st
        out = {}
        for (w, g), coefficient in (amod if anti else mod).act('L', 1, aw if anti else hw, e).items():
            if anti:
                key, factor = (hw, w, g), (-sp.I)**e/(-sp.I)**g
            else:
                key, factor = (w, aw, g), (-1)**(word_parity(aw)*(e ^ g))
            out[key] = sp.simplify(out.get(key, 0)+factor*coefficient)
        return out
    @lru_cache(None)
    def exponential(st, sign):
        vector = {st: sp.S.One}
        for anti in (False, True):
            acc, derivative = dict(vector), dict(vector)
            cutoff = int(word_level(st[1 if anti else 0]))
            for n in range(1, cutoff+1):
                nxt = {}
                for key, coefficient in derivative.items():
                    for dest, val in action(key, anti).items():
                        nxt[dest] = sp.simplify(nxt.get(dest, 0)+coefficient*val)
                for dest, val in nxt.items():
                    acc[dest] = sp.simplify(acc.get(dest, 0)+sign**n*val/sp.factorial(n))
                derivative = nxt
            vector = {key: val for key, val in acc.items() if val != 0}
        return vector
    rows = []
    for n, a in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)):
        basis, predicted = oracle.ramond_basis_gram(0, n, a)
        direct = sp.zeros(len(basis))
        for i, left in enumerate(basis):
            for j, right in enumerate(basis):
                value = sum(lc*rc*oracle.vertex(((), l[0], r[0]), ((), l[1], r[1]), l[2], r[2], 1)
                    for l, lc in exponential(left, 1).items() for r, rc in exponential(right, -1).items())
                direct[i, j] = sp.simplify((-1)**(n+a)*value)
        assert sp.simplify(direct-predicted) == sp.zeros(len(basis)), (n, a, direct-predicted)
        rows.append(dict(R_level=n, anti_R_level=a, dimension=len(basis),
                         exact_entry_checks=len(basis)**2))
        print(f'Independent BPZ two-point check: R levels {(n,a)}, dimension {len(basis)}.', flush=True)
    return rows


def block_coefficients(oracle, levels, anti=False):
    out = {}
    for label in CHANNELS:
        c, h, bs = ((sp.conjugate(oracle.anti_c), sp.conjugate(oracle.anti_h),
                    tuple(map(sp.conjugate, oracle.anti_betas))) if anti
                    else (oracle.c, oracle.h, oracle.betas))
        block = HumanNSRRThetaOracle(central_charge=c, h_ns=h, beta_r1=bs[0], beta_r2=bs[1],
                    form_parity=label[0], primary_parity=0, etas=label[1:])
        for level in levels:
            z = projected_components(block.coefficient_components(*level))
            out[level, label] = z.conjugate() if anti else z
    return {n: {label: out[n, label] for label in CHANNELS} for n in levels}


def descendant_checks():
    low = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1))
    mixed = ((3, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1),
             (0, 2, 0), (0, 0, 2), (2, 1, 0), (1, 2, 0), (5, 0, 0))
    fixtures = [(PhysicalNSRRBPZ(), list(product(low, repeat=2))
                  + [(n, m) for n in mixed for m in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]),
                (PhysicalNSRRBPZ(h=sp.Rational(11, 8), c=sp.Rational(143, 10),
                    beta2=3*sp.I/5, beta3=9*sp.I/10), list(product(low[:3], ((1, 1, 0), (0, 1, 1))))),
                (PhysicalNSRRBPZ(h=sp.Rational(4, 5)+sp.I/7, c=15+sp.I/5,
                    beta2=sp.Rational(1, 9)+2*sp.I/5, beta3=-sp.Rational(1, 8)+3*sp.I/5,
                    anti_h=sp.Rational(7, 8)-sp.I/11, anti_c=16-sp.I/6,
                    anti_beta2=sp.Rational(1, 6)+sp.I/3, anti_beta3=sp.Rational(1, 7)+sp.I/2),
                    list(product(low[:2], low[:2]))+[((0, 1, 0), (0, 0, 1)), ((1, 1, 0), (1, 0, 0))])]
    rows, series_rows = [], []
    for index, (oracle, pairs) in enumerate(fixtures):
        ns, ms = sorted({n for n, m in pairs}), sorted({m for n, m in pairs})
        f, ft = block_coefficients(oracle, ns), block_coefficients(oracle, ms, True)
        for n, m in pairs:
            for sign, rsign in product((1, -1), repeat=2):
                omega = (sign, rsign, 1)
                direct = oracle.coefficient(n, m, physical_lifts_slots=omega)
                for eta, etap in product((1, -1), repeat=2):
                    lc = (2 if eta == 1 else 0, 2 if eta == -1 else 0)
                    rc = (2 if etap == 1 else 0, 2 if etap == -1 else 0)
                    reduced = contract_nsrr_bilinear(descendant_blocks=f[n], antiholomorphic_blocks=ft[m],
                        left_bry=lc, right_bry=rc, physical_lifts_slots=omega,
                        primary=1, antiholomorphic_primary=1)['total']
                    actual = direct[eta, etap]
                    error = abs(actual-reduced)/max(1, abs(actual), abs(reduced))
                    assert error < 2e-11, (index, n, m, omega, eta, etap, actual, reduced)
                    rows.append(dict(fixture=index, levels=n, anti_levels=m, s=sign, r=rsign,
                        eta=eta, etap=etap, direct_real=actual.real, direct_imag=actual.imag,
                        reduced_real=reduced.real, reduced_imag=reduced.imag, scaled_error=error))
        # Match identical retained monomial pairs with fully independent
        # complex q's, pants coefficients, primary weights and log branches.
        lc, rc = (1.3+.4j, -.2+.7j), (.9-.1j, .6+.3j)
        q, qt = (.021+.012j, -.015+.008j, .033-.019j), (.018-.007j, .025+.011j, -.013+.018j)
        ph = cmath.exp((.8+.12j)*(cmath.log(q[0])+2j*math.pi)+(.9-.03j)*cmath.log(q[1])+1.1*cmath.log(q[2]))
        pa = cmath.exp((.7-.08j)*cmath.log(qt[0])+(.8+.04j)*cmath.log(qt[1])+1.05*(cmath.log(qt[2])-2j*math.pi))
        for sign, rsign in product((1, -1), repeat=2):
            direct_terms, reduced_terms = [], []
            for n, m in pairs:
                power = math.prod(q[k]**(n[k]/2 if k == 0 else n[k])
                                  *qt[k]**(m[k]/2 if k == 0 else m[k]) for k in range(3))
                ds = oracle.coefficient(n, m, physical_lifts_slots=(sign, rsign, 1))
                direct_terms.append(power*sum(lc[i]*rc[j]/4*ds[e, ep]
                    for i, e in enumerate((1, -1)) for j, ep in enumerate((1, -1))))
                reduced_terms.append(power*contract_nsrr_bilinear(descendant_blocks=f[n], antiholomorphic_blocks=ft[m],
                    left_bry=lc, right_bry=rc, physical_lifts_slots=(sign, rsign, 1),
                    primary=1, antiholomorphic_primary=1)['total'])
            direct, reduced = ph*pa*csum(direct_terms), ph*pa*csum(reduced_terms)
            ratio = reduced/direct
            assert abs(ratio-1) < 2e-11
            series_rows.append(dict(fixture=index, s=sign, r=rsign,
                norm_ratio=abs(ratio), phase_difference_rad=cmath.phase(ratio),
                direct=encode(direct), reduced=encode(reduced), primary=encode(ph), anti_primary=encode(pa)))
        print(f'Physical NSRR descendant fixture {index}: {len(pairs)*16} coefficient checks.', flush=True)
    return rows, series_rows


def torus_checks():
    """Identity degeneration after the explicit two-point BPZ frame change."""
    oracle = PhysicalNSRRBPZ(h=0, beta3=2*sp.I/5)
    # Coefficients of prod_(n>=1)(1+x^n)/(1-x^n).
    multiplicity = [1, 2, 4, 8]
    rows = []
    for n, a in product(range(4), range(2)):
        basis, gram = oracle.ramond_basis_gram(0, n, a)
        identity = gram.inv()*gram
        parity = sp.diag(*[(-1)**(word_parity(hw)+word_parity(aw)+e) for hw, aw, e in basis])
        ordinary, supertrace = sp.simplify(sp.trace(identity)), sp.simplify(sp.trace(parity*identity))
        assert ordinary == 2*multiplicity[n]*multiplicity[a] and supertrace == 0
        rows.append(dict(level=n, anti_level=a, ordinary_trace=int(ordinary), supertrace=int(supertrace)))
    # The genus-two reduced matrix has exactly the same identity ground
    # multiplicity: BRY E=2,O=0, F0=sqrt2,F1=-i sqrt2, Ft1=+i sqrt2.
    blocks = {label: math.sqrt(2)*(1 if label[0] == 0 else -1j) for label in CHANNELS}
    for sign, rsign in product((1, -1), repeat=2):
        value = contract_nsrr_bilinear(descendant_blocks=blocks,
            antiholomorphic_blocks={k: z.conjugate() for k, z in blocks.items()},
            left_bry=(2, 0), right_bry=(2, 0), physical_lifts_slots=(sign, rsign, 1),
            primary=1, antiholomorphic_primary=1)['total']
        assert abs(value-(2 if rsign == -1 else 0)) < 1e-13
    return dict(descendant_trace_rows=rows, matrix_identity_ground='2 for r=-1; 0 for r=+1',
        oscillator_trace='2 product_n (1+q^n)/(1-q^n) product_n (1+qtilde^n)/(1-qtilde^n)',
        primary='q^h_R qtilde^htilde_R outside trace; subtract c/24 only in the cylinder character convention',
        scope='Identity neck and two-point BPZ coordinate conversion; no finite-neck torus-modulus identification assumed.')


def free_norm_phase_checks(inputs):
    """NSRR c=3/2 check against unbosonized modes AND phased theta roots.

    The literal Human combinations mix the two even RR spin blocks. The
    transformation is fixed by NS parity, before comparing any numbers.
    This tests the PHASE of a single Majorana, not just its squared norm.
    """
    from nsrr_genus2_block import level_triples
    from audit_free_bosonization_phase import ramond_majorana_coefficients, ramond_value, radial_roots
    from physical_free_plumbing_resummation import theta_charged_boson_resummation
    from spin_structure import SpinCharacteristic
    sys.path.insert(0, str(ROOT/'Code/genus_2_cross_channel'))
    cfg = inputs.read(DATA/'nsrr_double_virasoro_N7_L5_20260911/config.json')
    original = tuple(map(complex, next(p for p in cfg['points'] if p['point_id']=='generic_04')['source']['q_values']))
    p0, p1 = sp.Rational(31, 100), sp.Rational(47, 100)
    fermion_coefficients = ramond_majorana_coefficients(6)
    spins = [SpinCharacteristic((1, 1), b) for b in ((0, 0), (1, 1))]
    references = {}
    for scale in (.02, .1):
        q = tuple(scale*z for z in original)
        roots, branch = radial_roots(q, spins, steps=24, max_mode=24)
        fock = [ramond_value(fermion_coefficients, q, 6, beta=spin.beta) for spin in spins]
        assert max(abs(x/y-1) for x, y in zip(fock, roots)) < 3e-11
        references[scale] = (q, roots, fock, branch)
    rows = []
    for eta in (1, -1):
        charge0, charge1 = p0, -eta*p1
        h = (charge0+charge1)**2/2
        oracle = HumanNSRRThetaOracle(central_charge=sp.Rational(3, 2), h_ns=h,
            beta_r1=sp.I*p1/sp.sqrt(2), beta_r2=sp.I*p0/sp.sqrt(2),
            form_parity=0, primary_parity=0, etas=(eta, eta))
        components = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                      for e in level_triples(6)}
        for scale, (q, roots, fock, branch) in references.items():
            qs = q[::-1]
            weights = (h, sp.Rational(1, 16)+p1*p1/2, sp.Rational(1, 16)+p0*p0/2)
            primary = cmath.exp(sum(float(w)*cmath.log(z) for w, z in zip(weights, qs)))
            even, odd = 0j, 0j
            for e, vector in components.items():
                value = projected_components(vector)*math.prod(qs[k]**(e[k]/2) for k in range(3))
                if e[0] % 2: odd += value
                else: even += value
            boson = complex(theta_charged_boson_resummation(q, alpha_zero=float(charge0),
                                    alpha_one=float(charge1), max_mode=24).chiral_value)
            for sign in (1, -1):
                # G_s = Fhat0+s Fhat1 = (1-is) even +(1+is) odd.
                actual = primary*((1-1j*sign)*even+(1+1j*sign)*odd)
                # even=(D00+D11)/2, odd=(D00-D11)/2, hence
                # G_s=D00-i*s*D11; its ground phase is 1-i*s.
                reference = boson*(roots[0]-1j*sign*roots[1])
                fock_reference = boson*(fock[0]-1j*sign*fock[1])
                ratio = actual/reference
                assert abs(ratio-1) < 5e-7, (eta, scale, sign, ratio)
                rows.append(dict(eta=eta, s=sign, q_scale=scale, descendant_cutoff=3,
                    norm_ratio=abs(ratio), phase_difference_rad=cmath.phase(ratio),
                    complex_relative_error=abs(ratio-1),
                    fock_bosonization_relative_error=abs(fock_reference/reference-1),
                    primary=encode(primary), chiral_value=encode(actual), bosonized_value=encode(reference),
                    theta_root_branch=branch))
        print(f'One-Majorana norm and phase check in the bilinear NSRR basis: eta={eta:+d}.', flush=True)
    return rows


def saved_comparison(inputs):
    """Recombine complete complex datasets for ALL four physical tube signs."""
    free = {p['t']: p for p in inputs.read(DATA/'fixed_spin_free_NSrr_20260830/summary.json')['points']}
    target = {r['t']: r for r in inputs.read(DATA/'nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json')['rows']
              if r['quadrature_order'] == 5 and r['recursion_order'] == 16}
    output, node_count, primary_error = [], 0, 0.
    for name in ('nsrr_factorized_sign_trial_L3_N5_20260830', 'nsrr_trial_L5_N3_local_20260830'):
        directory = DATA/name
        config = inputs.read(directory/'summary.json')['config']
        digest = object_digest(config)
        assert config['channels'] == [list(k) for k in CHANNELS]
        tasks = [(n, i) for n in config['quadrature_orders'] for i in range(n**3)]
        paths = sorted((directory/'shards').glob('node-*.json'))
        assert len(paths) == len(tasks)
        sums = {}
        for index, (path, task) in enumerate(zip(paths, tasks)):
            shard = inputs.read(path)
            node_count += 1
            assert (shard['index'], shard['quadrature_order'], shard['node'], shard['config_digest']) == (index, *task, digest)
            constants = tuple(decode(v) for v in shard['C_BRY'])
            rows = {(r['t'], float(r['level']), tuple(r['lifts_geometry'])): r for r in shard['rows']}
            assert len(rows) == len(shard['rows']) == len(config['points'])*len(config['levels'])*len(config['lifts_geometry'])
            for point in config['points']:
                t = point['t']
                assert point['q_geometry'] == free[t]['source_NSrr']['q_values']
                expected_p = NSRRPlumbingInputs(tuple(map(complex, point['q_geometry'])),
                    (1, 1, 1), GEOMETRY_SECTORS).primary(config['b'], shard['momenta_geometry'])
                for level in config['levels']:
                    pair = [rows[t, float(level), lift] for lift in BASE_PROJECTION_LIFTS]
                    primary = decode(pair[0]['primary'])
                    assert primary == decode(pair[1]['primary'])
                    primary_error = max(primary_error, relative(primary, expected_p))
                    blocks = {k: csum(decode(row['blocks'][i]) for row in pair)/math.sqrt(2)
                              for i, k in enumerate(CHANNELS)}
                    for sign, rsign in product((1, -1), repeat=2):
                        result = contract_nsrr_bilinear(descendant_blocks=blocks,
                            antiholomorphic_blocks={k: v.conjugate() for k, v in blocks.items()},
                            left_bry=constants, right_bry=constants,
                            physical_lifts_slots=(sign, rsign, 1), primary=primary,
                            antiholomorphic_primary=primary.conjugate())
                        key = (task[0], float(level), t, sign, rsign)
                        sums.setdefault(key, []).append(shard['measure']*result['total'])
        kappa = 1+2*(config['b']+1/config['b'])**2
        for (n, level, t, sign, rsign), terms in sorted(sums.items()):
            z = csum(terms)
            tq = target[t]['target_Z']/free[t]['target_NSnsns']['Z_free']**kappa
            # These denominators carry the SAVED marked spin. Retaining
            # them is a diagnostic, not a spin-transport certificate.
            sq = z/free[t]['source_NSrr']['Z_free']**kappa
            output.append(dict(dataset=name, momentum_order=n, source_order=level, t=t, s=sign, r=rsign,
                source_Z_real=z.real, source_Z_imag=z.imag, source_Q_real=sq.real, source_Q_imag=sq.imag,
                saved_target_Q=tq, ratio_norm=abs(sq/tq), ratio_phase_rad=cmath.phase(sq/tq)))
        print(f'Recombined all four tube-sign choices: {name}.', flush=True)
    assert primary_error < 2e-13
    return output, dict(input_nodes=node_count, maximum_primary_relative_error=primary_error,
        interpretation='Literal full-state Human theta tube signs. Ratios to the saved marked-spin denominator/target are diagnostic; no identification of the two spin frames is inferred from the chiral projection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--skip-saved', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  convention='physical NSRR BPZ bilinear; independent HJS anti data')
    result['clifford_derivation'] = clifford_derivation()
    result['independent_two_point_gram_checks'] = identity_gram_checks()
    rows, series = descendant_checks()
    write_csv(args.output/'descendant_checks.csv', rows)
    result['descendant_checks'] = dict(count=len(rows), maximum_scaled_error=max(r['scaled_error'] for r in rows))
    result['complex_series_checks'] = series
    result['genus_one'] = torus_checks()
    result['free_norm_phase_checks'] = free_norm_phase_checks(inputs)
    if not args.skip_saved:
        saved, provenance = saved_comparison(inputs)
        write_csv(args.output/'saved_comparison.csv', saved)
        result['saved_comparison'] = provenance
    for file in ('Code/genus_2/nsrr_bilinear_sewing.py', 'Code/genus_2/nsrr_bilinear_state_oracle.py',
                 'Code/genus_2/audit_nsrr_bilinear_sewing.py',
                 'Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py',
                 'Code/double_virasoro/nsrr/nsrr_genus2_block.py', 'Human Notes/SCblock.tex'):
        inputs.bytes(ROOT/file)
    (args.output/'provenance.json').write_text(json.dumps(inputs.files, indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: val for key, val in result.items() if key in ('descendant_checks', 'saved_comparison')}, indent=2))


if __name__ == '__main__':
    main()
