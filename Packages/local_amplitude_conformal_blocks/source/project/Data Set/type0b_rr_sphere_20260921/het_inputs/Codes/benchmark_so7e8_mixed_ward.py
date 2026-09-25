"""Independent PCO and original-exterior checks, before momentum integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from benchmark_so7e8_four_ramond_convergence import MOMENTA
from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel
from so7e8_two_ramond_liouville_integral import _swapped_to_original


def run(order=7, momenta=MOMENTA, internal_momentum=.37, *, channel="crossed",
        species_list=("SS","SV","VS","VV"), check_inversion=True,
        moduli=(.37+.23j,.68-.15j,.81+.08j)):
    started = time.perf_counter()
    if channel not in ("direct", "crossed"):
        raise ValueError("channel must be direct or crossed")
    u = np.asarray(moduli, dtype=complex)
    rows = {}
    for species in species_list:
        def kernel(m, s, picture):
            return build_fixed_p_two_ramond_kernel(internal_momentum,
                external_liouville_momenta=m,time_momenta=(*m[:3],-m[3]),
                ns_at_z=s[0],ns_at_one=s[1],zero_ramond_family="Psi_tilde",
                infinity_ramond_family="Psi",channel=channel,
                picture_zero_at=picture,maximum_twice_level=order,
                digits=max(100,4*order+40),
                crossed_block_backend="double_virasoro" if channel=="crossed" else "inverse_gram",
                direct_block_backend="double_virasoro",
                allow_uncertified_direct_projector=channel=="direct")
        k = kernel(momenta,species,"z")
        other = kernel(momenta,species,"one")
        a = np.array([list(k.evaluate(z).values()) for z in u])
        b = np.array([list(other.evaluate(z).values()) for z in u])
        inversion = None
        if check_inversion:
            m = momenta
            swapped = kernel((m[0],m[2],m[1],m[3]),species[::-1],"one")
            exterior = np.array([list(k.evaluate(1/z).values()) for z in u])/abs(u[:,None])**4
            folded = np.array([list(_swapped_to_original(swapped.evaluate(z),tuple(species)).values()) for z in u])
            inversion = (np.linalg.norm(exterior-folded,axis=1)/np.linalg.norm(exterior,axis=1)).tolist()
        pairs = lambda values: np.stack((values.real, values.imag), axis=-1).tolist()
        rows[species] = dict(
            picture_discrepancy=(np.linalg.norm(a-b,axis=1)/np.linalg.norm(a,axis=1)).tolist(),
            picture_discrepancy_relative_to_larger=(np.linalg.norm(a-b,axis=1)/
                np.maximum(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1))).tolist(),
            picture_z_values=pairs(a),picture_one_values=pairs(b),
            inversion_discrepancy=inversion)
    sources = (Path(__file__), Path(__file__).with_name("so7e8_ramond_sphere_integrand.py"),
               Path(__file__).with_name("so7e8_internal_ramond_double_virasoro.py"),
               Path(__file__).with_name("so7e8_mixed_double_virasoro.py"))
    return dict(order=order, internal_momentum=internal_momentum,
                channel=channel,backend="double_virasoro",full_pbw_block_used=False,
                momenta=[[p.real,p.imag] for p in momenta],
                moduli=[[z.real,z.imag] for z in u], species=rows,
                inversion_checked=check_inversion,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                seconds=time.perf_counter()-started, production_certified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[7])
    parser.add_argument("--channel", choices=("crossed", "direct"), default="crossed")
    parser.add_argument("--species", choices=("SS", "SV", "VS", "VV"), nargs="+", default=["VV"])
    parser.add_argument("--no-inversion", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = [run(order, channel=args.channel, species_list=args.species,
                  check_inversion=not args.no_inversion) for order in args.orders]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
