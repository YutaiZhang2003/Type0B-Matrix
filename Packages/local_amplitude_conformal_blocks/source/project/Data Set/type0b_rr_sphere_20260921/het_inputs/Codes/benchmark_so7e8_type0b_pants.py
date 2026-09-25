"""Compare local human-normalized forms and BRY data with Type0B sources.

The sibling repository is a read-only independent reference for this audit;
it is NOT a runtime dependency of the amplitude or coefficient generators.
"""
import argparse
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import G,L
from so7e8_human_conventions import human_ns_three_point, human_ns_pants, human_rr_pants_matrix
from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants


def run(root):
    root=Path(root).resolve()
    sources={
        "human_note": root/"Human Notes/SCblock.tex",
        "setup": root/"Machine Notes/type0b_matrix_model_setup.tex",
        "constants": root/"Code/c_Recursion/super_liouville_structure_constants.py",
        "ns_sign": root/"Code/c_Recursion/ns_human_convention.py",
        "ns_ward": root/"Code/c_Recursion/ns_genus2_symbolic_low_order.py",
    }
    spec=importlib.util.spec_from_file_location("_type0b_reference_constants",sources["constants"])
    reference=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rows=[]
    for momenta in ((.21,.38,.54),(-.21,.38,.54),(.21,.38,-.54),
                    (.21+.07j,.38-.03j,.54+.02j)):
        actual=(*ns_structure_constants(*momenta),*rr_ns_structure_constants(*momenta))
        expected=(reference.ns_structure_constant(*momenta),reference.ns_tilde_structure_constant(*momenta),
                  *reference.rr_ns_structure_constants(*momenta))
        residual=max(abs(a-e)/max(abs(e),1e-300) for a,e in zip(actual,expected))
        hn=human_ns_pants(*momenta)
        matrix=human_rr_pants_matrix(*momenta)
        encode=lambda x: [complex(x).real,complex(x).imag]
        rows.append(dict(momenta=list(map(encode,momenta)),bry_constants=list(map(encode,actual)),
                         relative_reference_defect=residual,human_ns_pants=list(map(encode,hn)),
                         human_rr_sign_matrix=[[encode(x) for x in row] for row in matrix]))
    sys.path.insert(0,str(sources["ns_ward"].parent))
    try:
        spec=importlib.util.spec_from_file_location("_type0b_reference_ns_ward",sources["ns_ward"])
        ward=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=ward
        spec.loader.exec_module(ward)
        h=(sp.Rational(3,7),sp.Rational(5,11),sp.Rational(7,13))
        c=sp.Rational(27,2)
        words=((),(G(-sp.Rational(1,2)),),(G(-sp.Rational(3,2)),),(L(-2),))
        bad=[]
        count=0
        for intrinsic in product((0,1),repeat=3):
            oracle=ward.ExactNSDescendantThreeForm(c=c,weights=h,primary_parities=intrinsic)
            for triple in product(words,repeat=3):
                actual=human_ns_three_point(*triple,h_infinity=h[0],h_middle=h[1],h_zero=h[2],
                                           c=c,primary_parities=intrinsic)
                expected=oracle.value(*[tuple((m.kind,int(2*m.index)) for m in w) for w in triple])
                residual=sp.cancel(actual-expected)
                count+=1
                if residual!=0:
                    bad.append(dict(intrinsic=intrinsic,words=str(triple),residual=str(residual)))
    finally:
        sys.path.pop(0)
    return dict(authority="Type0B Human Notes/SCblock.tex; unchanged",
        references={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for key,path in sources.items()},
        sources_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ("so7e8_human_conventions.py","spin23_super_liouville_data.py",
                                     "benchmark_so7e8_type0b_pants.py")},
        constants=rows,ns_ward=dict(comparisons=count,mismatches=len(bad),details=bad,
                                  scope="each leg in {1,G_-1/2,G_-3/2,L_-2}; all 8 intrinsic parities"),
        odd_ns_branch="+i, adopted from Type0B; square fixed by BPZ, branch not inferred from crossing",
        physical_projector_certified=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type0b-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.type0b_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
