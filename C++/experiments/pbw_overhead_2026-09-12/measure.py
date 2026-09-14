"""Time PBW metadata and arithmetic overhead only; no Gram/Ward/block evaluation."""
from fractions import Fraction
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Code/ramond_zero_mode_recovery'))
import complex_arithmetic as ca
import numpy as np
from check_level10_complex import load_backend


def main():
    start = time.perf_counter()
    ca.ctx.prec = 136
    backend = load_backend()
    ca.ctx.prec = 136
    states = [backend.metadata('NS', 20), backend.metadata('R', 10), backend.metadata('R', 10)]
    modules = [SimpleNamespace(sector=s) for s in ('NS', 'R', 'R')]
    word = backend.PBWModule.word
    unit_i, sqrt2 = ca.F(ca.acb(0, 1)), ca.F(ca.acb(2).sqrt())
    scalar = ca.F(ca.acb(11, 7) / 23)
    preparation_start = time.perf_counter()
    words = [[word(m, s) for s in ss] for m, ss in zip(modules, states)]
    phases = [((-1 + unit_i) / sqrt2) ** k for k in range(3)]
    preparation = time.perf_counter() - preparation_start
    trials = 10000
    selected = [(j % len(states[0]), (7*j) % len(states[1]), (13*j) % len(states[2]))
                for j in range(trials)]

    def entries(cached):
        checksum = 0
        tick = time.perf_counter()
        for i, j, k in selected:
            a, b, c = states[0][i], states[1][j], states[2][k]
            if cached:
                current = words[0][i], words[1][j], words[2][k]
                phase = phases[b[2]+c[2]]
            else:
                current = [word(m, s) for m, s in zip(modules, (a,b,c))]
                phase = ((-1 + unit_i) / sqrt2) ** (b[2]+c[2])
            # The same final phase multiplication; no form.value call.
            value = phase * scalar
            checksum += len(current[0]) + bool(value)
        return time.perf_counter()-tick, checksum

    records = {'precision_bits': 136, 'dps': 40,
               'new_physical_gram_ward_or_block_evaluations': 0,
               'metadata_entries_per_trial': trials,
               'metadata_cache_preparation_seconds': preparation,
               'metadata_trials': {'current': [], 'cached': []}, 'matrix_trials': []}
    checksums = set()
    for order in (False, True), (True, False), (False, True):
        for cached in order:
            elapsed, checksum = entries(cached)
            checksums.add(checksum)
            records['metadata_trials']['cached' if cached else 'current'].append(elapsed)
    assert len(checksums) == 1
    current = statistics.median(records['metadata_trials']['current'])
    cached = statistics.median(records['metadata_trials']['cached'])
    records['metadata_speedup'] = current/cached
    records['metadata_seconds_saved_per_entry'] = (current-cached)/trials
    print('Metadata:', records['metadata_speedup'], 'x', flush=True)

    for n, columns in ((32, 64), (161, 64), (232, 64), (232, 512)):
        samples = [ca.F(ca.acb(11+j, 7-j) / ca.acb(37+j, 19)) for j in range(64)]
        a = np.array([samples[(7*i+3*j)%64] for i in range(n) for j in range(n)], dtype=object).reshape(n,n)
        b = np.array([samples[(5*i+11*j)%64] for i in range(n) for j in range(columns)], dtype=object).reshape(n,columns)
        av, bv = ca.flint_matrix(a), ca.flint_matrix(b)
        row = {'shape': [n,n,columns], 'current': [], 'native_operands': []}
        for order in (False, True), (True, False), (False, True):
            for native in order:
                tick = time.perf_counter()
                result = av*bv if native else ca.mm(a,b)
                elapsed = time.perf_counter()-tick
                row['native_operands' if native else 'current'].append(elapsed)
                assert (result.nrows(), result.ncols()) == (n,columns) if native else result.shape == (n,columns)
        row['speedup'] = statistics.median(row['current'])/statistics.median(row['native_operands'])
        records['matrix_trials'].append(row)
        print('Matrix:', row['shape'], row['speedup'], 'x', flush=True)

    ns = [len(backend.metadata('NS',n)) for n in range(21)]
    ramond = [len(backend.metadata('R',n))//2 for n in range(11)]
    vertices = 4*sum(ns)*sum(ramond)**2
    records['box_counts'] = {
        'requested_tensor_entries': vertices,
        'current_word_constructions': 3*vertices,
        'cached_words_including_both_ramond_grounds_on_each_edge': sum(ns)+4*sum(ramond),
        'current_phase_exponentiations': vertices,
        'cached_phase_exponentiations': 3,
        'gram_entries': sum(d*d for d in ns)+4*sum(d*d for d in ramond),
        'triangular_gram_entries': sum(d*(d+1)//2 for d in ns)+4*sum(d*(d+1)//2 for d in ramond),
        'dense_contraction_products': sum(2*a*b*c*(a+b+c+1) for a in ns for b in ramond for c in ramond),
    }
    assert vertices == 1318075412
    records['metadata_only_box_savings_estimate_seconds'] = vertices*(current-cached)/trials
    records['limitations'] = [
        'These are synthetic component timings, not a PBW block profile or whole-block speedup.',
        'Metadata sample uses words at edge level ten; lower levels have shorter words.',
        'Native matrix timing excludes layout changes, precision-policy midpoint handling, and final extraction; it is an arithmetic-kernel target.',
        'No Ward recurrence optimization or compiled PBW implementation was benchmarked.',
        'No coefficient-accuracy conclusion follows from these timings.',
    ]
    records['wall_seconds'] = time.perf_counter()-start
    (HERE/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)


if __name__ == '__main__':
    main()
