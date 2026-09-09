#!/usr/bin/env python3
"""Reproducible component/PBW/c-recursion and exact h-pole audits.

The seed polynomials are extracted from PBW, explicitly not an independent
prediction.  The h-pole residue identities and the independent c-recursion
comparison test the component transport and reconstruction conventions.
"""

import argparse
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import sympy as sp

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'c_Recursion'))
from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from ns_pillow_direct_pbw import DirectNSSpherePBW, NSPrimaryWard
from ns_pillow_elliptic_audit import NSEllipticRecursion, indices
from ns_pillow_components import (ExactPBWSeeds, ComponentEllipticRecursion,
                                  InteriorUnitSeed, component_pullback, component_vertex_labels)


CASES = {
    'A': ('1.27',('.31','.42','.53','.37','.47','.28'),('.73','1.10','1.37')),
    'B': ('1.43',('.22','.61','.39','.58','.74','.45'),('1.13','.85','1.69')),
    'C': ('.83',('.17','.83','1.21','.46','.34','.67'),('.19','1.47','.82')),
    'D': ('1.61',('1.12','.26','.79','1.33','.58','.91'),('2.31','.64','1.09')),
}


def fixture(points,case,number):
    b,ds,hs = CASES[case]
    b = number(b)
    external = tuple(map(number,ds[:points-2]+ds[-2:]))
    internal = tuple(map(number,hs[:points-3]))
    return b,number('1.5')+3*(b+1/b)**2,external,internal


def audit_ward():
    c,h,d,k = sp.symbols('c h d k')
    ward = NSPrimaryWard(c,(h,d,k),True)
    expected = {(0,0,0):1,(1,0,0):1,(0,1,0):1,(0,0,1):-1,
                (1,1,0):h+d-k,(1,0,1):h-d+k,(0,1,1):h-d-k,
                (1,1,1):-(h+d+k-sp.Rational(1,2))}
    g=(('G',-1),)
    rows=[]
    for beta,reference in expected.items():
        actual=ward.value(g if beta[0] else (),beta[1],g if beta[2] else ())
        rows.append(dict(markings=beta,value=str(actual),exact_zero=sp.expand(actual-reference)==0))
    return dict(passed=all(row['exact_zero'] for row in rows),rows=rows)


def plane_audit(points,degree,case,markings,dps,unit_seed=False):
    b,c,ds,hs=fixture(points,case,mp.mpf)
    keys=tuple(indices(points-3,2*degree))
    rows=[]
    for beta in markings:
        direct=DirectNSSpherePBW(c,ds,hs,external_descendants=beta)
        other=NSSphereLinearCRecursion(central_charge=c,external_weights=ds,internal_weights=hs,
              external_descendants=beta,vertex_sectors=(sum(beta)%2,)+(0,)*(points-3),working_precision=dps)
        other._edge_residue=lru_cache(None)(other._edge_residue)
        other._global_coefficient=lru_cache(None)(other._global_coefficient)
        plane,c_plane={},{}
        for key in keys:
            plane[key]=direct.coefficient(key)
            c_plane[key]=other._coefficient(key,c,hs,component_vertex_labels(key,beta))
        left=component_pullback(plane,c,ds,hs,beta,degree)
        right=component_pullback(c_plane,c,ds,hs,beta,degree)
        error=max(abs(left.get(k,0)-right.get(k,0))/max(1,abs(left.get(k,0)),abs(right.get(k,0))) for k in keys)
        row=dict(markings=beta,coefficients=len(keys),maximum_scaled_error=mp.nstr(error,12),passed=error<mp.mpf('1e-'+str(dps//2)))
        if unit_seed:
            recursion=ComponentEllipticRecursion(b,ds,hs,beta,seed=InteriorUnitSeed(beta))
            h_error=max(abs(left.get(k,0)-recursion.coefficient(k))/max(1,abs(left.get(k,0)),abs(recursion.coefficient(k))) for k in keys)
            row.update(h_vs_pbw_maximum_scaled_error=mp.nstr(h_error,12),
                       passed=bool(row['passed'] and h_error<mp.mpf('1e-'+str(dps//2))),
                       h_seed='Independent proposed interior unit seed; no PBW or c coefficient input')
        rows.append(row)
        print('PBW/C',beta,'PASS' if row['passed'] else 'FAIL',row['maximum_scaled_error'],flush=True)
    return dict(passed=all(row['passed'] for row in rows),rows=rows,
                coefficient_count=sum(row['coefficients'] for row in rows),
                comparison='independent plane PBW and c-recursion, both pulled back with effective component weights')


def symbolic_audit(points,degree,case,markings,fully_generic=False):
    b,c,ds,hs=fixture(points,case,sp.Rational)
    if fully_generic:
        b=sp.Symbol('b')
        c=sp.Rational(3,2)+3*(b+1/b)**2
        ds=sp.symbols(f'd1:{points+1}')
    records=[]
    for beta in markings:
        start=time.monotonic()
        seeds=ExactPBWSeeds(c,ds,beta,degree).build()
        if fully_generic:
            print('SYMBOLIC PBW/SEEDS READY',beta,f'{time.monotonic()-start:.2f}s',flush=True)
        h=seeds.internal
        recursion=ComponentEllipticRecursion(b,ds,h,beta,seed=seeds,symbolic=True)
        fractions={key:sp.cancel(seeds._pulled.get(key,0)).as_numer_denom() for key in seeds.keys}
        checks=[]
        for key in seeds.keys:
            numerator,denominator=fractions[key]
            labels=component_vertex_labels(key,beta)
            poles=[(i,r,s) for i,n in enumerate(key) for r in range(1,n+1)
                   for s in range(1,n//r+1) if (r+s)%2==0]
            allowed=sp.prod(h[i]-recursion.pole(r,s) for i,r,s in poles)
            domain=sp.QQ.frac_field(b) if fully_generic else sp.QQ
            allowed_poly=sp.Poly(allowed,*h,domain=domain)
            denominator_poly=sp.Poly(denominator,*h,domain=domain)
            pole_ok=allowed_poly.rem(denominator_poly).is_zero
            residues=[]
            for edge,r,s in poles:
                pole=recursion.pole(r,s)
                residue=(numerator.subs(h[edge],pole)/sp.diff(denominator,h[edge]).subs(h[edge],pole)
                         if sp.cancel(denominator.subs(h[edge],pole))==0 else sp.S.Zero)
                left=(ds[0],ds[1]) if edge==0 else (h[edge-1],ds[edge+1])
                right=(ds[-1],ds[-2]) if edge==len(h)-1 else (h[edge+1],ds[edge+2])
                kernel=((-1)**(r*s)*recursion.norm_slope(r,s)
                        *recursion.fusion(r,s,labels[edge],*left)*recursion.fusion(r,s,labels[edge+1],*right))
                child_key=list(key);child_key[edge]-=r*s
                cn,cd=fractions[tuple(child_key)]
                child=(cn/cd).subs(h[edge],pole+sp.Rational(r*s,2))
                factor=4**(r*s) if len(h)==1 else (2**(r*s) if edge in (0,len(h)-1) else 1)
                # Cancel the rational pole substitution in each factor
                # before taking their product. Otherwise SymPy expands huge
                # powers of b-denominators which cancel identically later.
                residue=sp.cancel(residue)
                child=sp.cancel(child)
                expected=sp.cancel(factor*sp.cancel(kernel)*child)
                residues.append(dict(edge=edge,r=r,s=s,exact_zero=sp.cancel(residue-expected)==0))
            checks.append(dict(twice_levels=key,simple_pole_catalog_passed=bool(pole_ok),residues=residues,
                               regular_polynomial=str(seeds.expression(key)),
                               passed=bool(pole_ok and all(row['exact_zero'] for row in residues))))
            if fully_generic:
                print('SYMBOLIC COEFFICIENT',key,'PASS' if checks[-1]['passed'] else 'FAIL',
                      f'{time.monotonic()-start:.2f}s',flush=True)
        # Check the actual executable common-shift recursion at fresh numeric
        # internal weights, not the PBW source symbols or a fitted h value.
        numeric_checks=[]
        if not fully_generic:
            with mp.workdps(70):
                numeric_seed=seeds.numeric(dps=70)
                numeric_b=mp.mpf(str(b.p))/mp.mpf(str(b.q))
                numeric_d=tuple(mp.mpf(str(d.p))/mp.mpf(str(d.q)) for d in ds)
                numeric_c=mp.mpf('1.5')+3*(numeric_b+1/numeric_b)**2
                for base in ('.39','1.21','2.07'):
                    weights=tuple(mp.mpf(base)+mp.mpf('.23')*i for i in range(len(h)))
                    evaluator=ComponentEllipticRecursion(numeric_b,numeric_d,weights,beta,seed=numeric_seed)
                    direct=DirectNSSpherePBW(numeric_c,numeric_d,weights,external_descendants=beta)
                    plane={key:direct.coefficient(key) for key in seeds.keys}
                    reference=component_pullback(plane,numeric_c,numeric_d,weights,beta,degree)
                    maximum=max(abs(reference.get(k,0)-evaluator.coefficient(k))/max(1,abs(reference.get(k,0)),abs(evaluator.coefficient(k))) for k in seeds.keys)
                    numeric_checks.append(dict(internal_weights=list(map(str,weights)),maximum_scaled_error=mp.nstr(maximum,12),passed=maximum<mp.mpf('1e-45')))
        passed=all(row['passed'] for row in checks) and all(row['passed'] for row in numeric_checks)
        records.append(dict(markings=beta,passed=passed,coefficients=checks,numeric_common_shift_checks=numeric_checks,
                            seconds=time.monotonic()-start))
        print('H-POLES/SEEDS',beta,'PASS' if passed else 'FAIL',f'{time.monotonic()-start:.2f}s',flush=True)
    return dict(passed=all(row['passed'] for row in records),rows=records,
                coefficient_count=sum(len(row['coefficients']) for row in records),
                residue_count=sum(len(c['residues']) for row in records for c in row['coefficients']),
                symbolic_parameters=('b, all external and all internal weights' if fully_generic else
                                     'all internal weights; fixed exact rational b and external weights from case '+case),
                seed_origin='exact PBW polynomial division at common-weight infinity; not independent seed validation',
                seed_symbols={'base':'common_weight','differences':'seed_a2,...,seed_a_(n-3)'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('ward','plane','symbolic','unit-seed'),required=True)
    parser.add_argument('--points',type=int,choices=(4,5,6),default=5)
    parser.add_argument('--degree',type=int,default=3)
    parser.add_argument('--case',choices=tuple(CASES),default='A')
    parser.add_argument('--dps',type=int,default=80)
    parser.add_argument('--markings',help='one bit string, or omit to test all markings')
    parser.add_argument('--fully-generic',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.degree<0 or args.dps<30:
        parser.error('degree must be nonnegative and dps at least 30')
    if args.markings is not None and (len(args.markings)!=args.points or any(x not in '01' for x in args.markings)):
        parser.error('markings must contain exactly --points zero/one digits')
    markings=(tuple(map(int,args.markings)),) if args.markings else tuple(product((0,1),repeat=args.points))
    start=time.monotonic()
    with mp.workdps(args.dps):
        if args.mode=='ward':
            result=audit_ward()
        elif args.mode in ('plane','unit-seed'):
            result=plane_audit(args.points,args.degree,args.case,markings,args.dps,args.mode=='unit-seed')
        else:
            result=symbolic_audit(args.points,args.degree,args.case,markings,args.fully_generic)
    sources=[Path(__file__),Path(__file__).with_name('ns_pillow_components.py'),
             Path(__file__).with_name('ns_pillow_direct_pbw.py'),Path(__file__).with_name('ns_pillow_elliptic_audit.py'),
             CODE/'c_Recursion/ns_multipoint_c_recursion.py',CODE/'c_Recursion/ns_global_osp_block.py',
             CODE/'c_Recursion/ns_recursion_recipe.py',CODE/'c_Recursion/ns_human_convention.py']
    result.update(mode=args.mode,points=args.points,degree=args.degree,case=args.case,dps=args.dps,
                  fully_generic=args.fully_generic,seconds=time.monotonic()-start,
                  source_sha256={str(path.relative_to(CODE.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
