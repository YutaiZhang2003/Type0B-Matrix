#!/usr/bin/env python3
"""Freeze a reproducible Cannon crossing-gate release and amplitude specification."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from benchmark_so7e8_four_ramond_convergence import MOMENTA as PILOT, FAMILIES
from benchmark_so7e8_two_ramond_generic_scan import MOMENTA as EXISTING
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_literature_campaign import ROOT, atomic_json, digest, pairs
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule


def configuration(p_nodes=32):
    if not isinstance(p_nodes, int) or isinstance(p_nodes, bool) or p_nodes < 3:
        raise ValueError("at least three total momentum nodes per channel required")
    four_ns = json.loads((ROOT/"Codes/spin23_vvvv_threshold_settings.json").read_text())["production"]
    return dict(
        precision=four_ns["mp_dps"],
        blocks=dict(backend="literature double Virasoro; ordinary sphere CCY c-recursion",
                    baseline_order=4, maximum_order=10, relative_tolerance=.02,
                    order_unit="physical internal level / degree in q; stored degree in sqrt(q) is 2*order",
                    first_comparison=[4, 5], stop_at_cap="retain result, flag unconverged; never certify by cap alone"),
        extrapolation=dict(epsilon=.01, sample_b=[1.01, 1.02],
                           variable="Q-2=(b-1)^2/b",
                           formula="[4*(1+epsilon)*F(1+epsilon)-(1+2*epsilon)*F(1+2*epsilon)]/(3+2*epsilon)",
                           target="assembled native coefficient tables at fixed physical momenta; exact b=1 prefactors",
                           error_control="independent low-order b=1 Ward and Cauchy checks; not an automatic error bound"),
        momentum=dict(nodes_per_channel=p_nodes, budget="TOTAL per internal NS/R channel",
                      scheme="threshold_weighted", threshold_options=four_ns["momentum_threshold_options"],
                      sector_beta=dict(NS=2., R=0.), measure="dP; heterotic amplitude applies 1/pi once"),
        moduli=dict(source="Codes/spin23_vvvv_threshold_settings.json:production",
                    settings={k: v for k, v in four_ns.items()
                              if k not in ("q_order", "p_nodes", "momentum_scheme", "momentum_threshold_options")},
                    atlas="original/swapped bulk, zero disk and one lens; apply adjacent-order rule on actual integration points before production"),
        crossing=dict(points=pairs((.35+.1j, .5+.1j, .65+.1j)), relative_tolerance=.02,
                      tolerance_origin="preparation acceptance criterion; separate from the user's 2% block stopping rule",
                      integration="internal P only; no moduli integration",
                      components="all 64 mixed and 16 RRRR external physical components, including independent NS left/right superpartners",
                      branch="principal slit lifts; t=exp(i*pi*tau/2); crossed 1-z keeps its opposite imaginary part; explicit clockwise exchange/Mobius phase",
                      antiholomorphic="conjugate convention coefficients analytically; keep physical complex momenta fixed"),
        cluster=dict(account="yin_lab", partition="shared", cpus_per_task=1, memory="8G",
                     walltime="12:00:00", maximum_concurrency=128),
    )


def tasks(config):
    energies = {"pilot": PILOT, "independent": (.03+.12j, .04+.18j, .02+.26j, .09+.56j), **EXISTING}
    banks, amplitudes, rules = [], [], {}
    for sector in ("NS", "R"):
        options = dict(config["momentum"]["threshold_options"], beta=config["momentum"]["sector_beta"][sector])
        rule = threshold_weighted_rule(config["momentum"]["nodes_per_channel"], options)
        rules[sector] = dict(metadata=rule.metadata(), momenta=rule.momenta.tolist(), dP_weights=rule.weights.tolist())
    for name, m in energies.items():
        require_conservative_real_p_chamber(m[:3], m[3])
        for species in ("SS", "SV", "VS", "VV"):
            amplitudes.append(dict(energy=name, observable="NSNSRR", ns_species=species,
                                   original_order="R0,NSz,NS1,Rinfinity", momenta=pairs(m),
                                   status="held_until_same-convention_physical_assembly_and_crossing_are_verified"))
        amplitudes.append(dict(energy=name, observable="RRRR", families=FAMILIES, momenta=pairs(m),
                               status="held_until_same-convention_physical_assembly_and_crossing_are_verified"))
        # Preserve the physical assignment of R momenta (original slots 0,3).
        mixed = (m[0], m[3], m[2], m[1])
        for observable, family, p in (("mixed", "mixed_ns", mixed), ("rrrr", "rrrr", m)):
            for channel in ("s", "t"):
                fam = "mixed_r" if observable == "mixed" and channel == "t" else family
                pp = p if channel == "s" else (p[2], p[1], p[0], p[3])
                sector = "R" if fam == "mixed_r" else "NS"
                points = [complex(*z) for z in config["crossing"]["points"]]
                if channel == "t":
                    points = [1-z for z in points]
                for j, (P, w) in enumerate(zip(rules[sector]["momenta"], rules[sector]["dP_weights"])):
                    banks.append(dict(energy=name, observable=observable, channel=channel, family=fam,
                                      momenta=pairs(pp), points=pairs(points), sector=sector,
                                      momentum_index=j, P=P, dP_weight=w))
    return {name: pairs(p) for name, p in energies.items()}, banks, amplitudes, rules


def prepare(output, *, p_nodes=32):
    output = Path(output).resolve()
    if (output/"manifest.json").exists():
        raise FileExistsError("immutable release already exists; choose a new output directory")
    config = configuration(p_nodes)
    energies, banks, amplitudes, rules = tasks(config)
    files = sorted((ROOT/"Codes").rglob("*.py"))
    files += [ROOT/name for name in (
        "Codes/spin23_vvvv_threshold_settings.json", "cluster/submit_so7e8_literature_stage.slurm",
        "cluster/submit_so7e8_literature.py", "docs/so7e8/SO7E8_LITERATURE_CANNON_N32_20260920.md",
        "tests/test_literature_self_dual.py", "tests/test_so7e8_literature_campaign.py",
    )]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    type0b = Path("/Users/yutaizhang/Desktop/Type0B-Matrix/Code/two_loop/analysis/super_liouville_b1_extrapolation.py")
    manifest = dict(schema="so7e8-literature-ccy-cannon-v1", config=config,
                    energies=energies, momentum_rules=rules, bank_tasks=banks,
                    requested_amplitude_tasks=amplitudes, source_sha256=hashes,
                    extrapolation_provenance=dict(path=str(type0b), sha256=hashlib.sha256(type0b.read_bytes()).hexdigest()),
                    production_enabled=False,
                    release_scope="executable block banks and fixed-z crossing; amplitude specification reserved, old heterotic assembler disabled")
    manifest["manifest_sha256"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"manifest.json", manifest)
    with tarfile.open(output/"sources.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(output/"manifest.json", arcname="run/manifest.json")
        archive.add(type0b, arcname="references/type0b_duality_extrapolation.py")
    prepared = dict(status="prepared_not_submitted", executable_bank_tasks=len(banks),
                    requested_amplitude_tasks=len(amplitudes), heterotic_production_enabled=False,
                    manifest_sha256=manifest["manifest_sha256"],
                    archive_sha256=hashlib.sha256((output/"sources.tar.gz").read_bytes()).hexdigest())
    atomic_json(output/"prepared.json", prepared)
    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p-nodes", type=int, default=32,
                        help="total nodes per internal NS/R channel (current production choice: 32)")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, p_nodes=args.p_nodes), indent=2))
