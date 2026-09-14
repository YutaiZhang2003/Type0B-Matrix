"""Fresh independent-edge level-five PBW/DV timings and directed comparisons."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import argparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = '/private/tmp/theta_fermion_ccy_env/bin/python'
PBW = ROOT/'Code/theta_fermion_ccy/optimized_pbw.py'


def compare(a,b):
    left = {tuple(r['exponents']):r['values'] for r in a['coefficients']}
    right = {tuple(r['exponents']):r['values'] for r in b['coefficients']}
    assert left.keys() == right.keys()
    worst = Decimal(0)
    absolute = Decimal(0)
    location = None
    with localcontext() as ctx:
        ctx.prec = 65
        for n in left:
            for slot,(x,y) in enumerate(zip(left[n],right[n])):
                xr,xi,yr,yi = [Decimal(z) for z in (x['real'],x['imag'],y['real'],y['imag'])]
                assert all(z.is_finite() for z in (xr,xi,yr,yi))
                delta = ((xr-yr)**2+(xi-yi)**2).sqrt()
                scaled = delta/max(Decimal(1),(xr*xr+xi*xi).sqrt(),(yr*yr+yi*yi).sqrt())
                absolute = max(absolute,delta)
                if scaled > worst:
                    worst,location = scaled,{'exponents':n,'parity_slot':slot}
    return {'monomials':len(left),'parity_slots':8*len(left),
            'maximum_scaled_difference':str(worst),'maximum_absolute_difference':str(absolute),
            'worst_scaled_location':location,'criterion':'1e-20','passed':worst<Decimal('1e-20')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args = parser.parse_args()
    level = 2 if args.validate else 5
    directory = HERE/('validation' if args.validate else 'results')
    directory.mkdir(exist_ok=True)
    report = {'status':'running','precision_bits':136,'dps':40,'q_level_cutoffs':[level]*3,
              'environment':platform.platform(),
              'protocol':'sequential fresh processes; no numerical caches loaded; compilation excluded',
              'pbw_source_sha256':hashlib.sha256(PBW.read_bytes()).hexdigest(),'runs':[]}
    report_path = directory/'timings.json'
    algorithms = ['pbw_baseline','pbw_optimized'] if args.validate else ['pbw_optimized','double_virasoro']
    for algorithm in algorithms:
        for mode in ('ordinary','inserted'):
            stem = algorithm+'_'+mode
            output = directory/(stem+'.json')
            if algorithm.startswith('pbw'):
                command = [PYTHON,str(PBW),'--implementation',algorithm.split('_')[1]]
            else:
                command = [str(HERE/'ramond'),'--truncation','per-edge','--sector-policy','record']
            command += ['--mode',mode,'--level',str(level),'--dps','40','--json',str(output)]
            record = {'algorithm':algorithm,'mode':mode,'command':command,'status':'running'}
            report['runs'].append(record)
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            print('Starting',stem,'level',level,flush=True)
            tick = time.perf_counter()
            with (directory/(stem+'.log')).open('w') as log:
                process = subprocess.Popen(command,stdout=log,stderr=log)
                record['pid'] = process.pid
                report_path.write_text(json.dumps(report,indent=2)+'\n')
                try:
                    status = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            record['wall_seconds'] = time.perf_counter()-tick
            record['exit_code'] = status
            record['status'] = 'completed' if status == 0 else 'failed'
            if status:
                report['status'] = 'failed'
                report_path.write_text(json.dumps(report,indent=2)+'\n')
                print((directory/(stem+'.log')).read_text()[-4000:],flush=True)
                raise SystemExit(status)
            data = json.loads(output.read_text())
            record['timing_seconds'] = data['timing_seconds']
            record['counts'] = data['counts']
            record['coefficient_vectors'] = len(data['coefficients'])
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            print(stem,record['wall_seconds'],'s wall;',record['timing_seconds'],flush=True)
    report['comparisons'] = []
    for mode in ('ordinary','inserted'):
        inputs = [json.loads((directory/(algorithm+'_'+mode+'.json')).read_text()) for algorithm in algorithms]
        result = {'mode':mode,**compare(*inputs)}
        report['comparisons'].append(result)
        print(json.dumps(result),flush=True)
    report['status'] = 'completed' if all(r['passed'] for r in report['comparisons']) else 'comparison_failed'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if report['status'] != 'completed':
        raise SystemExit('Coefficient comparison failed; timings retained for diagnosis.')


if __name__ == '__main__':
    main()
