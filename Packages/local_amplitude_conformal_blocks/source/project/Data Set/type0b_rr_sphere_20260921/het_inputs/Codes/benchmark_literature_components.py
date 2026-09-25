#!/usr/bin/env python3
"""One fixed-coordinate parity-completeness check, with no moduli integral."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from literature_component_blocks import (ComponentBlocks, PaperConstants, SECTORS,
    crossing_phase, gauss_rule, sewn_integrand)


def run(order=16, p_order=16, z=.5+.1j):
    started = time.time()
    constants = PaperConstants(48)
    nodes,weights = gauss_rule([0,.25,.5,1,1.5,2,3,4,6],p_order)
    definitions = {
        'rrrr': ((.23,.41,.32,.47),list(product((0,1),repeat=4))),
        'mixed': ((.21,.39,.31,.43),[(r1,r2,(a3,b3),(a4,b4))
                   for r1,r2,a3,b3,a4,b4 in product((0,1),repeat=6)])}
    rows = []
    reflection_error = 0.
    for name,(p,cases) in definitions.items():
        sums = np.zeros((2,len(cases)),complex)
        for channel in (0,1):
            family = 'rrrr' if name == 'rrrr' else ('mixed_ns' if channel == 0 else 'mixed_r')
            pp = p if channel == 0 else (p[2],p[1],p[0],p[3])
            zz = z if channel == 0 else 1-z
            exts = cases if channel == 0 else [(e[2],e[1],e[0],e[3]) for e in cases]
            for P,w in zip(nodes,weights):
                blocks = ComponentBlocks(family,pp,P,order)
                for j,e in enumerate(exts):
                    sums[channel,j] += w*sewn_integrand(blocks,constants,P,zz,e)
            # Folding the full P contour must hold for the new components too.
            checks = []
            for P in (.71,-.71):
                blocks = ComponentBlocks(family,pp,P,min(order,8))
                checks.append(np.array([sewn_integrand(blocks,constants,P,zz,e) for e in exts]))
            reflection_error = max(reflection_error,float(np.max(abs(checks[0]-checks[1]))/max(np.max(abs(checks[0])),1e-300)))
            print(f'{family}, channel {channel}: {time.time()-started:.1f}s',flush=True)
        reference = max(np.max(abs(sums)),1e-300)
        for j,e in enumerate(cases):
            sectors = SECTORS['rrrr' if name == 'rrrr' else 'mixed_ns']
            total_parity = sum(v if s == 'R' else sum(v) for s,v in zip(sectors,e))%2
            phase = crossing_phase(sectors,e)
            lhs,rhs = sums[0,j],phase*sums[1,j]
            scale = max(abs(lhs),abs(rhs),reference*1e-12)
            pair = lambda x: [float(x.real),float(x.imag)]
            rows.append(dict(family=name,external=e,vanishes_by_parity=bool(total_parity),
                             crossing_phase=pair(phase),lhs=pair(lhs),rhs=pair(rhs),
                             relative_error=float(abs(lhs-rhs)/scale),
                             zero_residual=float(max(abs(lhs),abs(rhs))/reference) if total_parity else None))
    module = Path(__file__).with_name('literature_component_blocks.py')
    return dict(stage='literature conventions; no Human Note conversion',c=3,
                source=['https://arxiv.org/abs/1012.2974v2','https://arxiv.org/abs/0810.1203v2'],
                momenta={key:val[0] for key,val in definitions.items()},
                z=[complex(z).real,complex(z).imag],maximum_twice_level=order,
                method='literal three-point Ward identities and Gram sewing; finite exact substitution into elliptic coordinate',
                p_nodes=len(nodes),p_order=p_order,p_max=6.,upsilon_order=48,
                integration='internal P only; no moduli integration',
                branch='principal slit plane; clockwise z -> 1-z rotation, opposite antiholomorphic lift',
                reflection_error=reflection_error,elapsed_seconds=time.time()-started,
                code_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,default=16)
    parser.add_argument('--p-order',type=int,default=16)
    parser.add_argument('--z',type=complex,default=.5+.1j)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.order,args.p_order,args.z)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for family in ('rrrr','mixed'):
        print(family,'max error',max(x['relative_error'] for x in result['rows'] if x['family']==family),flush=True)
    print('P reflection:',result['reflection_error'],flush=True)
