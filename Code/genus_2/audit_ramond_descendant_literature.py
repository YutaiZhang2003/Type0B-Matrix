#!/usr/bin/env python3
"""Further Ramond descendant, dual-state, torus and genus-two checks.

References: Hadasz--Jaskolski--Suchanek arXiv:1207.5740, section 2.2 and
Appendix C; Suchanek arXiv:1012.2974, section 2. Primary q^h factors never
enter a descendant coefficient or a Gram/basis transformation here.
"""
from __future__ import annotations

import argparse
import cmath
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/path) for path in (
    "Code/c_Recursion", "Code/h_recursion", "Code/double_virasoro/nsrr",
    "Code/genus_2", "Code")]
from ramond_descendant_blocks import RamondVermaModule, RamondThreePointWardMatrix
from ramond_pbw_generalized_ward import RamondPBWModule, RamondState, GeneralizedNRRWard, word_parity
from ns_pbw_basis import ns_pbw_basis
from nsrr_genus2_block import HumanNSRRThetaOracle
from nsrr_reflected_state_sewing import ReflectedNSRRStateSewing

OUTPUT = ROOT/"Data Set/ramond_descendant_literature_20260915"


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def scaled_error(actual, expected):
    return float(abs(actual-expected)/max(1,abs(actual),abs(expected)))


def numeric(matrix):
    return np.asarray(matrix.evalf(30).tolist(),complex)


def source_hashes():
    paths = ("Code/h_recursion/ramond_descendant_blocks.py",
             "Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py",
             "Code/double_virasoro/nsrr/nsrr_genus2_block.py",
             "Code/genus_2/nsrr_reflected_state_sewing.py",
             "Code/c_Recursion/ns_pbw_basis.py",
             "Code/genus_2/audit_ramond_descendant_literature.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def gram_checks():
    """Two independently implemented algebras, both R parities through L=5."""
    rows = []
    for fixture,(c,beta) in enumerate(((s.Rational(81,5),s.I*s.Rational(7,10)),
                                     (s.Rational(149,4),s.I*s.Rational(9,10)))):
        h = c/24-beta**2
        exact = RamondPBWModule(h,beta,c)
        reference = RamondVermaModule(c=complex(c),weight=complex(h))
        a = beta*(1+s.I)/s.sqrt(2)  # G0 w+ = a w-.
        for level,parity in itertools.product(range(6),(0,1)):
            basis, gram = exact.gram_matrix(level,parity)
            old = reference.basis(level,parity)
            transform = s.zeros(len(basis))  # e_G0 = e_w * transform (columns).
            for j,word in enumerate(old):
                ground = int(("G",0) in word)
                state = RamondState(tuple((kind,s.Integer(n)) for kind,n in word if n),ground)
                transform[basis.index(state),j] = a**ground
            converted = (transform.T*gram*transform).applyfunc(s.simplify)
            ref_gram = np.asarray(reference.gram_matrix(level,parity),complex)
            actual = numeric(converted)
            errors = [scaled_error(x,y) for x,y in zip(actual.flat,ref_gram.flat)]
            hermitian = s.diag(*[(-s.I)**state.ground for state in basis])*gram
            assert (hermitian-hermitian.conjugate().T).applyfunc(s.simplify) == s.zeros(len(basis))
            reflected = (transform.conjugate().T*hermitian*transform-converted).applyfunc(s.simplify)
            assert reflected == s.zeros(len(basis))
            assert max(errors) < 5e-12, (fixture,level,parity,max(errors))
            minimum = float(np.linalg.eigvalsh(numeric(hermitian)).min())
            assert minimum > 0
            # This scalar is an exact identity in the computed algebra, not
            # a floating-point comparison with a fitted normalization.
            for state in basis:
                once = exact.act("G",s.S.Zero,state.word,state.ground)
                twice = {}
                for (word,g),coefficient in once.items():
                    for key,value in exact.act("G",s.S.Zero,word,g).items():
                        twice[key] = twice.get(key,s.S.Zero)+coefficient*value
                twice[(state.word,state.ground)] = twice.get((state.word,state.ground),0)-(h+level-c/24)
                assert all(s.simplify(value)==0 for value in twice.values())
            rows.append(dict(fixture=fixture,level=level,parity=parity,dimension=len(basis),
                gram_entries=len(basis)**2,max_scaled_error=max(errors),
                minimum_hermitian_eigenvalue=minimum,zero_mode_square_states=len(basis)))
        print(f"Gram and reflected-dual checks: fixture {fixture}, levels 0--5 passed.",flush=True)
    return dict(rows=rows,gram_entries=sum(row["gram_entries"] for row in rows),
                zero_mode_square_states=sum(row["zero_mode_square_states"] for row in rows),
                max_scaled_error=max(row["max_scaled_error"] for row in rows),
                exact_reflection_checks=len(rows),
                dictionary="e_G0=e_w*T; B_G0=T^T*B_w*T; H_w=diag((-i)^ground)*B_w; H_G0=T^dagger*H_w*T=B_G0 on imaginary beta")


def hjs_torus_coefficients(b,h,d,sign):
    """Literal independent transcription of arXiv:1207.5740 Appendix C.

    F_{beta,e}^{lambda(+-),n}, n=0,1,2. These are local descendant
    coefficients, not the character-stripped elliptic H coefficients.
    """
    denominator1 = (3+6*b*b+16*h)*(6+b*b*(3+16*h))
    denominator2 = denominator1*(11+30*b*b+16*h)*(30+b*b*(11+16*h))
    if sign == 1:
        first = (6*(2+5*b*b+2*b**4)*(3+4*(d-1)*d)
                 +64*(3+3*b**4+b*b*(3-6*d+2*d*d))*h+512*b*b*h*h)/denominator1
        second = 4*(180*(1+b**8)*(11+2*(d-1)*d+16*h)*(3+4*(d-1)*d+16*h)
            +b**4*(15*(3855+2*d*(-3077+5*d*(1021+4*d*(-124+37*d))))
                    +32*(6078+d*(-8961+d*(9083+224*(-12+d)*d)))*h
                    +512*(779+d*(-337+d*(353+4*(-12+d)*d)))*h*h
                    +8192*(14+d*(-11+5*d))*h**3+65536*h**4)
            +12*(b*b+b**6)*(16*d**4*(59+24*h)-8*d**3*(361+496*h)
                +8*d*d*(823+128*h*(13+7*h))+(13+48*h)*(213+32*h*(7+8*h))
                -32*d*(136+21*h*(17+16*h))))/denominator2
    elif sign == -1:
        first = (32*(4*(d-2)*d+3)*h*b*b+6*(2*b**4+5*b*b+2)*(4*d*d-1))/denominator1
        second = 4*(-1+2*d)*(180*(1+b**8)*(1+2*d)*(1+2*d*d+8*h)
            +12*(b*b+b**6)*(3*(39+88*h)+4*(d*(89+2*d*(-33+59*d))
                +4*d*(149+2*d*(-47+6*d))*h+64*(-3+8*d)*h*h))
            +b**4*(4*d**3*(2775+128*h*(7+2*h))-2*d*d*(4725+2432*h*(7+2*h))
                -9*(-305+8*h*(31+32*h*(1+8*h)))
                +6*d*(1525+8*h*(1047+32*h*(39+8*h)))))/denominator2
    else:
        raise ValueError("sign must be +/-1")
    return (1,first,second)


def ramond_character(order):
    """One parity's oscillator character prod_n (1+q^n)/(1-q^n)."""
    coefficients = [1]+[0]*order
    for n in range(1,order+1):
        next_values = coefficients[:]
        for k in range(n,order+1):
            next_values[k] += next_values[k-n]
        coefficients = [next_values[k]+(next_values[k-n] if k>=n else 0) for k in range(order+1)]
    return coefficients


def torus_checks():
    rows, trace_rows = [], []
    fixtures = ((1.1,.73,.83),(1.27,1.11,.42),(1.2,.83+.17j,.61-.13j))
    for fixture,(b,p,d) in enumerate(fixtures):
        c=1.5+3*(b+1/b)**2
        h=c/24+p*p
        module=RamondVermaModule(c=c,weight=h)
        for sign in (1,-1):
            vertex=RamondThreePointWardMatrix(left_module=module,right_module=module,external_ns_weight=d,sign=sign)
            expected=hjs_torus_coefficients(b,h,d,sign)
            for n in range(3):
                gram=np.asarray(module.gram_matrix(n,0))
                inserted=np.asarray(vertex.matrix(n,n,0))
                actual=np.trace(np.linalg.solve(gram,inserted))
                error=scaled_error(actual,expected[n])
                assert error < 2e-12, (fixture,sign,n,actual,expected[n],error)
                a,before=complex(actual),complex(expected[n])
                rows.append(dict(fixture=fixture,sign=sign,level=n,actual=pair(a),literature=pair(before),
                    scaled_error=error,norm_error=abs(abs(a)-abs(before)),
                    phase_error_radians=abs(cmath.phase(a/before)) if a and before else None))
    # Identity is the plus vertex. Check the actual inserted matrices,
    # rather than assuming that an inverse-Gram trace counts basis states.
    c=16.2;h=c/24+.7**2
    module=RamondVermaModule(c=c,weight=h)
    identity=RamondThreePointWardMatrix(left_module=module,right_module=module,external_ns_weight=0,sign=1)
    character=ramond_character(6)
    for n in range(7):
        traces=[]
        for parity in (0,1):
            gram=np.asarray(module.gram_matrix(n,parity))
            inserted=np.asarray(identity.matrix(n,n,parity))
            entry_error=max(scaled_error(a,b) for a,b in zip(gram.flat,inserted.flat))
            trace=np.trace(np.linalg.solve(gram,inserted))
            assert entry_error<1e-11 and abs(trace-character[n])<1e-9,(n,parity,entry_error,trace)
            traces.append(complex(trace))
            trace_rows.append(dict(level=n,parity=parity,trace=pair(trace),expected=character[n],max_matrix_error=entry_error))
        assert abs(traces[0]-traces[1])<1e-9
    return dict(rows=rows,identity_rows=trace_rows,literature_coefficients=len(rows),
                max_scaled_error=max(row["scaled_error"] for row in rows),
                max_phase_error_radians=max(row["phase_error_radians"] or 0 for row in rows),
                character_per_parity=character,full_long_module_character=[2*n for n in character],
                graded_long_module_character=[0]*7,
                primary="All coefficients are descendant-only. HJS q^(h-c/24) is external; the saved plumbing convention keeps q^h external and tracks cylinder factors separately.")


def ward_checks():
    """Extend the Clifford and reflected-bra identities to R level three.

    Both eta and form-parity choices, all ground labels, NS descendants
    through level two, total chiral level at most four.
    """
    c=s.Rational(81,5);h=s.Rational(7,10)
    beta2,beta3=s.I*s.Rational(2,5),s.I*s.Rational(7,10)
    forms={(f,t):GeneralizedNRRWard(p_phi=0,form_parity=f,eta=t,h_ns=h,
        h_second=c/24-beta2**2,h_third=c/24-beta3**2,
        beta_second=beta2,beta_third=beta3,central_charge=c)
        for f,t in itertools.product((0,1),(1,-1))}
    ns={n:[tuple((kind,s.Rational(k,2)) for kind,k in word) for word in ns_pbw_basis(n)] for n in range(5)}
    r={n:[state.word for state in RamondPBWModule.basis(n,0)] for n in range(4)}
    u=(1-s.I)/s.sqrt(2);j=s.Matrix([[0,u],[s.conjugate(u),0]])
    counts=dict(J_middle=0,J_ket=0,outgoing_bra=0)
    nonzero=0
    for nn,n2,n3 in itertools.product(range(5),range(4),range(4)):
        if nn+2*n2+2*n3>8:
            continue
        for f,t,a,x,y,b,g in itertools.product((0,1),(1,-1),ns[nn],r[n2],r[n3],(0,1),(0,1)):
            value=forms[f,t].value(a,x,b,y,g)
            other=forms[1-f,t].value(a,x,b,y,g)
            p2=(word_parity(x)+b)%2;p3=(word_parity(y)+g)%2
            lhs2=(-1)**word_parity(x)*j[1-b,b]*forms[f,t].value(a,x,1-b,y,g)
            lhs3=(-1)**word_parity(y)*j[1-g,g]*forms[f,t].value(a,x,b,y,1-g)
            errors=(lhs2-s.I**f*t*s.conjugate(u)*(-1)**p2*other,
                    lhs3-s.I**f*u*other,
                    s.I**(b+g)*s.conjugate(value)-(-s.I)**f*(-1)**p3*value)
            assert all(s.simplify(error)==0 for error in errors),(nn,n2,n3,f,t,a,x,y,b,g,errors)
            for name in counts:counts[name]+=1
            nonzero+=int(value!=0)
        print(f"Ward/dual identities passed at levels ({nn}/2,{n2},{n3}).",flush=True)
    return dict(exact_checks=counts,nonzero_vertex_values=nonzero,
                scope="Even NS highest state; NS descendant level <=2, each R level <=3, total chiral level <=4; both forms, both eta signs and all ground labels.",
                outgoing_identity="i^(ground2+ground3)*conjugate(rho_f)=(-i)^f*(-1)^full_ket_parity*rho_f")


def sewing_checks():
    """New physical-family contractions beyond the previous R-level-one check."""
    pairs=[((0,2,0),(0,0,0)),((0,0,2),(0,0,0)),
           ((1,2,0),(1,0,0)),((1,0,2),(1,0,0)),
           ((0,2,0),(0,1,0)),((0,0,2),(0,0,1)),
           ((0,2,0),(0,2,0)),((0,0,2),(0,0,2)),
           ((0,3,0),(0,0,0)),((0,0,3),(0,0,0)),
           ((0,1,1),(0,1,1)),((1,1,1),(1,0,0)),
           ((2,1,0),(0,0,1)),((2,0,1),(0,1,0)),
           ((0,2,1),(0,0,0)),((0,1,2),(0,0,0))]
    rows=[]
    fixtures=((s.Rational(7,10),s.Rational(81,5),s.I*s.Rational(2,5),s.I*s.Rational(7,10)),
              (s.Rational(11,8),s.Rational(143,10),s.I*s.Rational(3,5),s.I*s.Rational(9,10)))
    for fixture,(h,c,b2,b3) in enumerate(fixtures):
        oracle=ReflectedNSRRStateSewing(h=h,c=c,beta2=b2,beta3=b3)
        chiral={eta:HumanNSRRThetaOracle(central_charge=c,h_ns=h,beta_r1=b2,beta_r2=b3,
                    form_parity=0,primary_parity=0,etas=(eta,eta)) for eta in (1,-1)}
        @lru_cache(None)
        def block(eta,levels):
            values=chiral[eta].coefficient_components(*levels)
            return math.sqrt(2)*sum(z for parity,z in enumerate(values) if not parity&2)
        for levels,anti_levels in pairs:
            direct=oracle.coefficient(levels,anti_levels)
            for eta,etap in itertools.product((1,-1),repeat=2):
                expected=block(eta,levels)*block(etap,anti_levels).conjugate() if eta==etap else 0j
                actual=direct[eta,etap]
                error=scaled_error(actual,expected)
                assert error<1e-10,(fixture,levels,anti_levels,eta,etap,actual,expected,error)
                rows.append(dict(fixture=fixture,levels=levels,anti_levels=anti_levels,
                    eta_left=eta,eta_right=etap,direct=pair(actual),reduced=pair(expected),scaled_error=error))
            print(f"Physical sewing passed: fixture {fixture}, {levels} x {anti_levels}.",flush=True)
    return dict(rows=rows,coefficients=len(rows),max_scaled_error=max(row["scaled_error"] for row in rows),
                constants="Independent c_L,eta*c_R,eta-prime checked separately; c_+=E/2,c_-=O/2. No fitting of EE,EO,OE,OO products.",
                scope="Selected new full-state coefficients with R level 2 or 3 and mixed left/right descendants; supports the existing local reflected M, not global spin transport.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section",choices=("gram","torus","ward","sewing","all"),default="all")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    functions={"gram":gram_checks,"torus":torus_checks,"ward":ward_checks,"sewing":sewing_checks}
    for name, function in functions.items():
        if args.section not in (name,"all"):
            continue
        result=function()
        result["source_sha256"]=source_hashes()
        result["created_at_utc"]=datetime.now(timezone.utc).isoformat()
        (args.output/(name+".json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({key:value for key,value in result.items() if key not in ("rows","identity_rows","source_sha256")},indent=2),flush=True)


if __name__=="__main__":
    main()
