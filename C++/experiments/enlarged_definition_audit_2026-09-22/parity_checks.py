"""Exact local sewing-phase comparison; no conformal blocks are recomputed."""
from itertools import product
import json
from pathlib import Path

count = 0
for A, epsilon2, p, f, auxiliary1, auxiliary2 in product((0, 1), repeat=6):
    auxiliary3 = auxiliary1 ^ auxiliary2
    # Eq. (2.9) times its explicitly prescribed vertex weight in (2.10).
    displayed = 1j ** ((A + auxiliary1) % 2) * (-1j) ** A
    displayed *= (-1) ** (
        f * (A + epsilon2) + A * auxiliary1 + (epsilon2 + p) * auxiliary3
    )
    # Factors separately retained by the physical/fermion graph contraction.
    epsilon3 = (f + A + epsilon2) % 2
    implemented = 1j ** auxiliary1
    implemented *= (-1) ** (
        f * (1 + epsilon3) + p * auxiliary3 + epsilon2 * auxiliary3
    )
    assert displayed == implemented
    # Two equal theta vertices: only (-1)^mathsfA survives locally.
    edge_metrics = (-1) ** (auxiliary1 + auxiliary2 + auxiliary3)
    assert edge_metrics * displayed ** 2 == (-1) ** auxiliary1
    count += 1

result = {"local_cases": count, "theta_cases": count, "failed": 0}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
