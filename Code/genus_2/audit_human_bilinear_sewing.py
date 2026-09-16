"""Independent finite-state audit of the Human-Note bilinear convention."""
from __future__ import annotations

from itertools import product
from datetime import datetime, timezone
from functools import lru_cache
import cmath
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("Code", "Code/genus_2", "Code/c_Recursion", "Code/double_virasoro/nsrr"):
    sys.path.insert(0, str(ROOT / relative))

import numpy as np
import sympy as s

from human_bilinear_sewing import (
    all_ns_theta_matrix, contract_bilinear, graded_product_gram,
    pairing_in_new_bases, descendant_basis_change, nsrr_tensor_product_matrix,
)
from ns_genus2_symbolic_low_order import ExactDirectThetaOracle, ExactNSDescendantThreeForm


def quadratic(bits):
    return sum(bits[i]*bits[j] for i in range(3) for j in range(i+1, 3)) % 2


def all_ns_states():
    fixtures = [
        (s.Rational(81, 5), (s.Rational(7, 10), s.Rational(9, 10), s.Rational(11, 10)),
         s.Rational(143, 10), (s.Rational(3, 5), s.Rational(4, 5), s.Rational(6, 5))),
        (s.Rational(81, 5)+s.I/7, (s.Rational(7, 10)+s.I/11, s.Rational(9, 10), s.Rational(11, 10)),
         s.Rational(143, 10)-s.I/13, (s.Rational(3, 5), s.Rational(4, 5)-s.I/17, s.Rational(6, 5))),
    ]
    levels = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),
              (2,0,0),(3,0,0),(0,3,0),(2,1,0)]
    left = (1.1+.3j, -.2+.7j)
    right = (.4-.9j, 1.2+.1j)
    matrix = all_ns_theta_matrix(left, right)
    rows = []
    phase_rows = []
    for fixture, (c, h, ct, ht) in enumerate(fixtures):
        holo = ExactDirectThetaOracle(c=c, weights=h)
        anti = ExactDirectThetaOracle(c=ct, weights=ht)
        hf = ExactNSDescendantThreeForm(c=c, weights=h)
        af = ExactNSDescendantThreeForm(c=ct, weights=ht)
        direct_coefficients = {}
        for n, m in product(levels, repeat=2):
            if sum(n) % 2 != sum(m) % 2:
                continue
            a = sum(n) % 2
            hb = [holo.modules[i].basis(n[i]) for i in range(3)]
            ab = [anti.modules[i].basis(m[i]) for i in range(3)]
            bases = [list(product(hb[i], ab[i])) for i in range(3)]
            inverses = []
            for i in range(3):
                gram = graded_product_gram(
                    np.asarray(holo.modules[i].gram_matrix(n[i]), complex),
                    np.asarray(anti.modules[i].gram_matrix(m[i]), complex),
                    [n[i] % 2]*len(hb[i]), [m[i] % 2]*len(ab[i]))
                inverses.append(np.linalg.inv(gram))
            tensor = np.empty(tuple(map(len, bases)), complex)
            for indices in product(*(range(len(b)) for b in bases)):
                states = [bases[i][indices[i]] for i in range(3)]
                tensor[indices] = complex(hf.value(*(z[0] for z in states))
                                          * af.value(*(z[1] for z in states)))
            # The two local vertex reorderings occur twice. The full-state
            # theta sign is separate and is retained before factorization.
            full_parities = tuple((x+y) % 2 for x,y in zip(n,m))
            direct = ((-1)**quadratic(full_parities) * left[a]*right[a]
                      * np.einsum('ijk,il,jm,kn,lmn->', tensor, *inverses, tensor))
            f = [0j, 0j]; ft = [0j, 0j]
            f[a] = complex(holo.coefficient(n))
            ft[a] = complex(anti.coefficient(m))
            assembled = contract_bilinear(matrix=matrix, holomorphic_blocks=f,
                                          antiholomorphic_blocks=ft)
            error = abs(assembled-direct) / max(1, abs(direct), abs(assembled))
            assert error < 2e-12, (fixture, n, m, assembled, direct)
            direct_coefficients[n,m] = direct
            rows.append(dict(fixture=fixture, holomorphic_twice_levels=n,
                             antiholomorphic_twice_levels=m, scaled_error=error))
        # Four independent NS tube lifts on each side, complex plumbing,
        # and unequal continued primary factors. Compare complex amplitudes,
        # not only squared norms. Slot order is (infinity,one,zero).
        lifts = tuple((1,a,b) for a,b in product((1,-1), repeat=2))
        logs = (cmath.log(.013+.007j)+2j*cmath.pi, cmath.log(.021-.005j), cmath.log(.017+.009j))
        alogs = (cmath.log(.019-.003j), cmath.log(.015+.008j)-2j*cmath.pi, cmath.log(.011-.006j))
        primary = cmath.exp(sum(complex(w)*ell for w,ell in zip(h,logs)))
        aprimary = cmath.exp(sum(complex(w)*ell for w,ell in zip(ht,alogs)))
        for lift, alift in product(lifts, repeat=2):
            monomials = {n: cmath.exp(sum(n[e]*logs[e]/2 for e in range(3)))
                        * np.prod([lift[e]**n[e] for e in range(3)]) for n in levels}
            amonomials = {n: cmath.exp(sum(n[e]*alogs[e]/2 for e in range(3)))
                         * np.prod([alift[e]**n[e] for e in range(3)]) for n in levels}
            f = [sum(complex(holo.coefficient(n))*monomials[n] for n in levels if sum(n)%2==a) for a in (0,1)]
            ft = [sum(complex(anti.coefficient(n))*amonomials[n] for n in levels if sum(n)%2==a) for a in (0,1)]
            assembled = contract_bilinear(matrix=matrix, holomorphic_blocks=f, antiholomorphic_blocks=ft,
                                          holomorphic_primary=primary, antiholomorphic_primary=aprimary)
            direct = primary*aprimary*sum(z*monomials[n]*amonomials[m]
                                          for (n,m),z in direct_coefficients.items())
            ratio = assembled/direct
            assert abs(ratio-1) < 3e-12
            phase_rows.append(dict(fixture=fixture, holomorphic_lifts=lift, antiholomorphic_lifts=alift,
                                   norm_ratio=abs(ratio), phase_difference_rad=cmath.phase(ratio),
                                   complex_relative_error=abs(ratio-1)))
    return dict(checks=len(rows), maximum_scaled_error=max(r['scaled_error'] for r in rows),
                norm_phase_checks=len(phase_rows),
                maximum_norm_ratio_error=max(abs(r['norm_ratio']-1) for r in phase_rows),
                maximum_phase_difference_rad=max(abs(r['phase_difference_rad']) for r in phase_rows),
                norm_phase_rows=phase_rows, rows=rows)


def parity_and_basis_checks():
    bits = list(product((0,1), repeat=3))
    count = 0
    for n, nt, p, pt in product(bits, repeat=4):
        x = tuple(a^b for a,b in zip(n,p))
        y = tuple(a^b for a,b in zip(nt,pt))
        if sum(x) % 2 != sum(y) % 2:
            continue
        original = (quadratic(tuple(a^b for a,b in zip(x,y)))+sum(a*b for a,b in zip(x,y))) % 2
        expected = (quadratic(x)+quadratic(y)+sum(x)) % 2
        assert original == expected
        count += 1
    m = np.asarray([[1+.2j, -.4j], [.3+.1j, 2-.7j]])
    u = np.asarray([[1, .2j], [.3, 1.1-.2j]])
    v = np.asarray([[.9+.1j, -.2], [.4j, 1.3]])
    f = np.asarray([.6+.2j, -.3+.5j])
    ft = np.asarray([-.1+.7j, .9-.4j])
    transformed = pairing_in_new_bases(m, u, v)
    before = contract_bilinear(matrix=m, holomorphic_blocks=f, antiholomorphic_blocks=ft)
    after = contract_bilinear(matrix=transformed, holomorphic_blocks=u@f, antiholomorphic_blocks=v@ft)
    assert abs(after-before) < 2e-14
    linear = contract_bilinear(matrix=m, holomorphic_blocks=1j*f, antiholomorphic_blocks=ft)
    assert abs(linear-1j*before) < 2e-14
    old_p = np.exp(np.asarray([.7+.2j, 1.1-.1j]) * np.asarray([-3+.4j, -4-.2j]))
    new_p = np.exp(np.asarray([1.3-.4j, .4+.1j]) * np.asarray([-2.5+.1j, -3+.6j]))
    old_pt = np.exp(np.asarray([.8-.1j, .6+.3j]) * np.asarray([-2-.3j, -4+.2j]))
    new_pt = np.exp(np.asarray([.2+.2j, 1.2-.3j]) * np.asarray([-3+.7j, -2.5-.2j]))
    fp = descendant_basis_change(u, old_p, new_p) @ f
    ftp = descendant_basis_change(v, old_pt, new_pt) @ ft
    propagated_before = contract_bilinear(
        matrix=m, holomorphic_blocks=f, antiholomorphic_blocks=ft,
        holomorphic_primary=old_p, antiholomorphic_primary=old_pt)
    propagated_after = contract_bilinear(
        matrix=transformed, holomorphic_blocks=fp, antiholomorphic_blocks=ftp,
        holomorphic_primary=new_p, antiholomorphic_primary=new_pt)
    assert abs(propagated_after-propagated_before) < 2e-14
    return dict(exact_grading_identities=count, complex_basis_error=abs(after-before),
                complex_linearity_error=abs(linear-1j*before),
                unequal_primary_transport_error=abs(propagated_after-propagated_before))


def unrestricted_nsrr_states():
    """Check the general tensor formula, WITHOUT a physical R projection.

    Independent complex coefficients t_L and t_R specify the vertex. This
    verifies the reduction to M, not the map (E,O) -> physical vertex t.
    """
    from nsrr_genus2_block import HumanNSRRThetaOracle

    labels = tuple(product((0,1), (1,-1), (1,-1)))
    left = {k: (.2+.13*j) + (.4-.07*j)*1j for j,k in enumerate(labels)}
    right = {k: (-.3+.09*j) + (.1+.11*j)*1j for j,k in enumerate(labels)}
    out_labels, matrix = nsrr_tensor_product_matrix(left, right)
    assert out_labels == labels
    fixtures = [(s.Rational(81,5), s.Rational(7,10), s.I*2/5, s.I*7/10),
                (s.Rational(143,10), s.Rational(9,10), s.I*3/5, s.I*9/10)]
    oracles = [{k: HumanNSRRThetaOracle(central_charge=c, h_ns=h,
                  beta_r1=b1, beta_r2=b2, form_parity=k[0], primary_parity=0,
                  etas=k[1:]) for k in labels} for c,h,b1,b2 in fixtures]

    @lru_cache(None)
    def data(side, levels):
        base = oracles[side][labels[0]]
        nb, ni = base.ns_basis_inverse(levels[0])
        bases = [nb]; grams = [np.linalg.inv(ni)]
        parities = [[levels[0] % 2]*len(nb)]
        for edge in range(2):
            module = base.r_modules[edge]
            rb = module.basis(levels[edge+1])
            bases.append(rb)
            grams.append(np.asarray([[complex(module.inner_product(x,y)) for y in rb] for x in rb]))
            parities.append([x.parity for x in rb])
        tensors = {}
        for f,t in product((0,1), (1,-1)):
            form = oracles[side][f,t,t].forms[0]
            tensor = np.empty(tuple(map(len,bases)), complex)
            for i,j,k in product(*(range(len(b)) for b in bases)):
                x,y,z = bases[0][i],bases[1][j],bases[2][k]
                tensor[i,j,k] = complex(form.value(base._ns_word(x), y.word,y.ground,z.word,z.ground))
            tensors[f,t] = tensor
        blocks = np.asarray([sum(oracles[side][k].coefficient_components(*levels)) for k in labels])
        return bases, grams, parities, tensors, blocks

    levels = [(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,0,1),(1,1,0)]
    rows = []
    for n,m in product(levels, repeat=2):
        hb,hg,hp,ht,hf = data(0,n)
        ab,ag,ap,at,af = data(1,m)
        inverse = [np.linalg.inv(graded_product_gram(hg[e],ag[e],hp[e],ap[e])) for e in range(3)]
        pairs = [list(product(range(len(hb[e])), range(len(ab[e])))) for e in range(3)]
        vl = np.empty(tuple(map(len,pairs)), complex); vr = np.empty_like(vl)
        topology = np.empty_like(vl)
        for indices in product(*(range(len(p)) for p in pairs)):
            hs, ats = zip(*(pairs[e][indices[e]] for e in range(3)))
            for tensor, coefficients in ((vl,left),(vr,right)):
                tensor[indices] = sum(coefficients[f,t,u]*ht[f,t][hs]*at[f,u][ats] for f,t,u in labels)
            parity = tuple(hp[e][hs[e]] ^ ap[e][ats[e]] for e in range(3))
            topology[indices] = (-1)**quadratic(parity)
        direct = np.einsum('ijk,il,jm,kn,lmn->', topology*vl, *inverse, vr, optimize=True)
        assembled = contract_bilinear(matrix=matrix, holomorphic_blocks=hf, antiholomorphic_blocks=af)
        error = abs(assembled-direct)/max(1,abs(assembled),abs(direct))
        assert error < 3e-12, (n,m,direct,assembled,error)
        rows.append(dict(holomorphic_levels=n, antiholomorphic_levels=m, scaled_error=error))
    return dict(checks=len(rows), maximum_scaled_error=max(r['scaled_error'] for r in rows),
                scope='Unrestricted tensor-product Ramond modules; no physical small-representation claim', rows=rows)


def ramond_dual_obstruction():
    """A Hermitian ket embedding is not automatically its bilinear dual."""
    i = s.I
    ket = s.Matrix([[1,0],[0,1],[0,1],[-i,0]]) / s.sqrt(2)
    # Suchanek's antiholomorphic ground phases give Btilde=diag(1,-i).
    # The full field's product includes the Human-Note crossing sign.
    gram = s.diag(1, -i, i, -1)
    same_embedding = s.simplify(ket.T*gram*ket)
    assert same_embedding == s.diag(1, 0)
    # This explicitly displayed dual is one ground-frame choice, not a
    # derivation of the descendant vertex/spin transport on the full graph.
    dual = s.Matrix([[1,0],[0,i],[0,-i],[-i,0]]) / s.sqrt(2)
    assert s.simplify(dual.T*gram*ket) == s.eye(2)
    # Put the antiholomorphic basis in the same algebraic ground-metric
    # convention as the holomorphic basis: wtilde^- = i*wbar^-.
    # Its canonical beta is then -beta. Transform coordinates as well as B.
    change = s.diag(1,i,1,i)
    canonical_gram = change.T*gram*change
    canonical_ket = change.inv()*ket
    canonical_dual = change.inv()*dual
    assert canonical_gram == s.diag(1,i,i,1)
    assert s.simplify(canonical_ket.T*canonical_gram*canonical_ket) == same_embedding
    assert s.simplify(canonical_dual.T*canonical_gram*canonical_ket) == s.eye(2)
    return dict(hjs_graded_product_gram=str(gram), hjs_ket_embedding=str(ket),
                canonical_graded_product_gram=str(canonical_gram),
                canonical_ket_embedding=str(canonical_ket),
                same_embedding_bilinear_gram=str(same_embedding),
                same_embedding_rank=same_embedding.rank(), ground_dual_example=str(canonical_dual),
                dual_pairing='identity',
                scope='Ground-state obstruction only; does not determine the interacting NSRR M or its global spin transport')


def main():
    out = ROOT / 'Data Set/human_bilinear_sewing_20260915'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(convention='Human-Note graded bilinear, not a Hermitian substitution',
                  all_ns=all_ns_states(), algebra=parity_and_basis_checks(),
                  unrestricted_nsrr=unrestricted_nsrr_states(),
                  ramond_ground=ramond_dual_obstruction())
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    paths = ['Human Notes/SCblock.tex', 'Code/genus_2/human_bilinear_sewing.py',
             'Code/genus_2/audit_human_bilinear_sewing.py',
             'Code/genus_2/compare_nsrr_nsnsns_theta.py',
             'Code/genus_2/test_human_bilinear_sewing.py',
             'Code/c_Recursion/ns_genus2_symbolic_low_order.py',
             'Code/double_virasoro/nsrr/nsrr_genus2_block.py',
             'Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: ({a:b for a,b in v.items() if not a.endswith('rows')} if isinstance(v,dict) else v)
                      for k,v in report.items() if k!='source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
