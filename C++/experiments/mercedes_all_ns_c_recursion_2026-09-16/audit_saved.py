"""Audit the saved comparison; no conformal-block calculation is rerun."""
from collections import Counter
from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(name, field):
    result = {}
    for line in (ROOT / name).read_text().splitlines():
        row = json.loads(line)
        key = row["case"], tuple(row["level2"])
        assert key not in result, ("duplicate", key)
        result[key] = {
            parity: (Decimal(real), Decimal(imag))
            for parity, real, imag in row[field]
        }
    return result


def norm(z):
    return (z[0] * z[0] + z[1] * z[1]).sqrt()


def main():
    dv = read("coefficients_L6.jsonl", "double_virasoro")
    ns = read("c_recursion_L6.jsonl", "c_recursion")
    expected = {k for k in product(range(13), repeat=6) if sum(k) <= 12}
    assert {key[1] for key in dv} == expected
    assert {key[1] for key in ns} == expected
    assert set(dv) == set(ns)
    slots = ((0, 1, 2), (0, 3, 5), (1, 4, 3), (2, 5, 4))
    maximum = {}
    counts = Counter()
    cases = Counter()
    zero = (Decimal(0), Decimal(0))
    with localcontext() as ctx:
        ctx.prec = 60
        for (case, levels), row in ns.items():
            bits = [x % 2 for x in levels]
            parity = sum(b << e for e, b in enumerate(bits))
            sector = sum((sum(bits[e] for e in slots[v]) % 2) << (v - 1)
                         for v in range(1, 4))
            assert case == sector
            assert set(row) <= {parity}
            assert set(dv[case, levels]) <= {parity}
            x, y = dv[case, levels].get(parity, zero), row.get(parity, zero)
            error = norm((x[0] - y[0], x[1] - y[1]))
            scaled = error / max(Decimal(1), norm(x), norm(y))
            assert scaled < Decimal("1e-18"), (case, levels, scaled)
            shell = sum(levels)
            old = maximum.setdefault(shell, [Decimal(0), Decimal(0)])
            old[0] = max(old[0], error)
            old[1] = max(old[1], scaled)
            counts[shell] += 1
            cases[case] += 1
    coverage = {
        "passed": True,
        "scope": "Saved coefficient audit only; no block or PBW rerun",
        "multidegrees": len(expected),
        "case_multidegrees": dict(sorted(cases.items())),
        "shells": [{"total_level": n / 2, "multidegrees": counts[n],
                    "max_absolute": str(maximum[n][0]),
                    "max_scaled": str(maximum[n][1])} for n in sorted(counts)],
    }
    (ROOT / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    paths = list(ROOT.glob("*.cpp")) + list(ROOT.glob("*.hpp"))
    paths += [ROOT / "Makefile", ROOT / "README.md", Path(__file__).resolve()]
    paths += list((ROOT.parent / "total_level6_2026-09-16").glob("*.hpp"))
    paths += [ROOT.parent / "total_level6_2026-09-16" / x
              for x in ("graph_validate.cpp", "pole_fragment.inc")]
    paths += list((REPO / "C++" / "include" / "ramond").glob("*.hpp"))
    paths += [REPO / "Human Notes" / "SCblock.tex"]
    paths += [ROOT / x for x in ("results_L6.json", "coefficients_L6.jsonl",
                               "c_recursion_L6.jsonl", "schottky_seed_L6.jsonl",
                               "coverage.json", "level6.log")]
    manifest = {
        "description": "40-digit all-NS Mercedes, double Virasoro vs NS c-recursion, total level six",
        "fresh_run": "./compare 6 > level6.log 2>&1",
        "sources_and_results_sha256": {
            str(path.relative_to(REPO)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passed": True, "multidegrees": len(expected),
                      "case_multidegrees": dict(sorted(cases.items()))}))


if __name__ == "__main__":
    main()
