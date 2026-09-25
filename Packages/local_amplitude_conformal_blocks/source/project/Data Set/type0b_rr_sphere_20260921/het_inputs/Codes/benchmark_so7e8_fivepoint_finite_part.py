"""Diagnostic only: resolve the endpoint/tail b=1 contour failure.

Failed contour tables may be recorded for comparison but are NEVER accepted
as quadrature nodes. The production 1e-7 criterion is not relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from so7e8_fivepoint_blocks import double_virasoro_fivepoint
from so7e8_fivepoint_primary import primary_labels,PERMUTATIONS,REAL_MOMENTA
from spin23_genus1_recursion import dictionary_finite_part


def pairs(z):
    return [complex(z).real,complex(z).imag]


def run(channel,output):
    started=time.perf_counter()
    external=tuple(REAL_MOMENTA[i] for i in PERMUTATIONS[channel])
    internal=(.03803847577293367,4.065703516359435)
    labels=primary_labels(channel)
    first={label:double_virasoro_fivepoint(channel=channel,b=.83,internal_momenta=internal,
                  external_momenta=external,maximum_twice_levels=8,
                  edge_parities=label[0],structure_signs=label[1]) for label in labels}
    keys=tuple((p,s,i,j) for (p,s),source in first.items() for i,j in source.coefficients)
    def evaluate(b):
        values={}
        for p,s in labels:
            series=double_virasoro_fivepoint(channel=channel,b=b,internal_momenta=internal,
                external_momenta=external,maximum_twice_levels=8,edge_parities=p,structure_signs=s)
            values.update({(p,s,i,j):v for (i,j),v in series.coefficients.items()})
        return values
    report=dict(channel=channel,internal_momenta=internal,rows=[],production_tolerance=1e-7,
                source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in (Path(__file__).name,'so7e8_fivepoint_blocks.py',
                                 'virasoro_fivepoint_c_recursion.py','spin23_genus1_recursion.py')},
                diagnostic_only=True,accepted_integral_node=False)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    previous=None
    for radius,check,samples in ((.10,.12,32),(.10,.12,48),(.08,.10,48),(.10,.12,64)):
        values,diags=dictionary_finite_part(evaluate,keys=keys,radius=radius,check_radius=check,samples=samples)
        errors={k:d.absolute_error/max(1.,abs(d.value),abs(d.check_value)) for k,d in diags.items()}
        worst=max(errors,key=errors.get)
        row=dict(radius=radius,check_radius=check,samples=samples,maximum_radius_defect=errors[worst],
            worst_key=str(worst),worst_value=pairs(values[worst]),worst_check_value=pairs(diags[worst].check_value),
            change_from_previous=None if previous is None else max(abs(values[k]-previous[k])/max(1.,abs(values[k]),abs(previous[k])) for k in keys),
            contour_passes=errors[worst]<1e-7,
            coefficients=[dict(key=str(k),value=pairs(v)) for k,v in values.items()])
        report['rows'].append(row)
        report['seconds']=time.perf_counter()-started
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(channel,{k:v for k,v in row.items() if k!='coefficients'},flush=True)
        previous=values
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',choices=('B','C'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.channel,args.output)
