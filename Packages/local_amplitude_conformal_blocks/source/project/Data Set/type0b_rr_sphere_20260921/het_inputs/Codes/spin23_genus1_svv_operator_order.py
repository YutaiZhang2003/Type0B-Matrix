"""Koszul signs for the ordered zero-picture SVV vertices.

At each vertex the holomorphic factor precedes the antiholomorphic factor.
Factor the product into vertex-ordered nonchiral SL fields, then time
fermions, then spectator fermions. Moving the parity insertion from edge
zero to the trace closure is a separate, compensating operation.
"""
from itertools import combinations


def grouping_sign(selected_time_indices):
    selected=set(selected_time_indices);tokens=[]
    for i in range(3):
        tokens.append(('time',i) if i in selected else ('sl_h',i))
        tokens.append(('sl_a',i) if i==0 else ('spectator',i))
    def target_key(token):
        kind,i=token
        return (0,i,kind=='sl_a') if kind.startswith('sl_') else (1 if kind=='time' else 2,i,0)
    keys=list(map(target_key,tokens))
    inversions=sum(keys[i]>keys[j] for i in range(len(keys)) for j in range(i+1,len(keys)))
    return (-1)**inversions


def assembly_factors(spin):
    if spin not in ('NS','NS_tilde','R'):raise ValueError('even spin structure required')
    lift=-1 if spin=='NS_tilde' else 1
    selected_sets=((),)+tuple(combinations(range(3),2))
    # Vertex zero contains bar-G V. Its SL parity is odd precisely when
    # the holomorphic factor was taken from the time-fermion sector.
    return tuple(grouping_sign(selected)*lift**int(0 in selected) for selected in selected_sets)


def canonicalize_record(record):
    """Apply exact ordering/cut corrections to a saved modular-test ledger."""
    def c(z):return complex(z['real'],z['imag'])
    evaluations=record['evaluations']
    original=[c(z)*f for z,f in zip(evaluations[0]['components'],assembly_factors(evaluations[0]['spin']))]
    transformed=[c(z)*f for z,f in zip(evaluations[1]['components'],assembly_factors(evaluations[1]['spin']))]
    weight=abs(c(evaluations[0]['tau']))**8
    return dict(original=original,transformed=transformed,
        total_ratio=sum(transformed)/(weight*sum(original)),
        individual_ratios=[transformed[i]/(weight*original[j]) for i,j in [(0,0),(1,2),(2,1),(3,3)]])
