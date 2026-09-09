import json
from pathlib import Path
import tempfile
import unittest

import mpmath as mp

from genus0_ns_elliptic_h_recursion import (
    compute_h_recursion, load_table, KacPoleError, total_degree_indices,
    coordinates_from_segment_nomes, invert_aligned_coordinates,
    effective_plumbing_parameters, ns_pillow_product,
    reconstruct_from_real_moduli, reconstruct_from_segment_nomes,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGERS = ROOT/"validation/Data Set/h-Recursion"


def fixture(points=6,case="A"):
    return json.loads((LEDGERS/f"ns_pillow_c_comparison_n{points}_{case}_degree10_dps80_20260830.json").read_text())


def compute(data,order=3,dps=60,**extra):
    return compute_h_recursion(b=data["b"],external_weights=data["external"],
            internal_weights=data["internal"],order=order,dps=dps,**extra)


class RuntimeTests(unittest.TestCase):
    def test_port_matches_independent_c_and_pbw_all_generic_fixtures_degree_three(self):
        with mp.workdps(70):
            for points,case in [(4,"A")]+[(n,k) for n in (5,6) for k in "ABCD"]:
                data = fixture(points,case)
                table = compute(data)
                rows = {tuple(r["twice_levels"]):r for r in data["coefficients"]}
                for key,value in table.coefficients.items():
                    for name in ("c_recursion_H","h_recursion_H","pbw_H"):
                        reference = mp.mpc(rows[key][name]["real"],rows[key][name]["imag"])
                        self.assertLess(abs(value-reference)/max(1,abs(reference)),mp.mpf("1e-46"))

    def test_counts_and_general_n_smoke(self):
        self.assertEqual([len(list(total_degree_indices(m,10))) for m in (1,2,3)],[21,231,1771])
        table = compute_h_recursion(b="1.27",external_weights=(".1",".2",".3",".4",".5",".6",".7"),
                     internal_weights=(".73","1.1","1.37","1.63"),order=2,dps=50)
        self.assertEqual(len(table.coefficients),70)
        self.assertEqual(len(table.evaluate_sectors((".01",)*4)),16)

    def test_parity_selection_and_default_is_not_a_sector_sum(self):
        table = compute(fixture(5))
        p = (".01",".02")
        sectors = table.evaluate_sectors(p)
        self.assertEqual(table.evaluate(p),sectors[(0,0)])
        self.assertNotEqual(table.evaluate(p),mp.fsum(sectors.values()))
        self.assertEqual(table.evaluate(p,parity=(1,1),order="0.5"),0)
        with mp.workdps(table.dps):
            self.assertLess(abs(table.shell(p,"0.5",parity=(1,0))+mp.sqrt(mp.mpf(".01"))/mp.mpf(".73")),mp.mpf("1e-55"))

    def test_odd_sheet_lift(self):
        table = compute(fixture(5))
        with mp.workdps(70):
            p = tuple(map(mp.mpf,(".01",".02")))
            logs = (mp.log(p[0])+2j*mp.pi,mp.log(p[1]))
            normal = table.evaluate(p,parity=(1,0))
            lifted = table.evaluate(p,parity=(1,0),log_nomes=logs)
            self.assertLess(abs(normal+lifted),mp.mpf("1e-55"))
            with self.assertRaises(ValueError):
                table.evaluate((-.01,.02),parity=(1,0))
            with self.assertRaises(ValueError):
                table.evaluate(p,log_nomes=(0,0))

    def test_roundtrip_and_invalid_table(self):
        table = compute(fixture(5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"table.json"
            table.save(path)
            loaded = load_table(path)
            self.assertEqual(loaded.coefficients,table.coefficients)
            self.assertEqual(loaded.evaluate((".1",".2"),parity=(1,0)),table.evaluate((".1",".2"),parity=(1,0)))
            broken = json.loads(path.read_text()); broken["coefficients"].pop()
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                load_table(path)

    def test_c_input_and_b_inverse_duality(self):
        data = fixture(5,"C")
        left = compute(data)
        with mp.workdps(70):
            right = compute_h_recursion(central_charge=left.central_charge,
                        external_weights=data["external"],internal_weights=data["internal"],order=3,dps=60)
            for key in left.coefficients:
                self.assertLess(abs(left.coefficients[key]-right.coefficients[key]),mp.mpf("1e-49"))

    def test_poles_and_input_guards(self):
        common = dict(external_weights=(".3",".4",".5",".6"),internal_weights=(".7",),order=2,dps=50)
        with self.assertRaises(KacPoleError):
            compute_h_recursion(b="1",**common)
        bad = dict(common); bad["internal_weights"]=("0",)
        with self.assertRaises(KacPoleError):
            compute_h_recursion(b="1.27",**bad)
        for extra in ({"b":"1.27","central_charge":"14"},{},{"b":"nan"}):
            with self.assertRaises(ValueError):
                compute_h_recursion(**common,**extra)
        for name,value in (("order",-1),("order",True),("dps",15)):
            bad = dict(common); bad[name]=value
            with self.assertRaises(ValueError):
                compute_h_recursion(b="1.27",**bad)

    def test_geometry_and_full_reconstruction(self):
        table = compute(fixture())
        with mp.workdps(60):
            inverse = invert_aligned_coordinates(".08",(".30",".65"),dps=60)
            z,mobiles = coordinates_from_segment_nomes(inverse.segment_nomes)
            for actual,expected in zip((z,)+mobiles,map(mp.mpf,(".08",".30",".65"))):
                self.assertLess(abs(actual-expected),mp.mpf("1e-49"))
            first = reconstruct_from_real_moduli(table,z=".08",mobile_positions=(".30",".65"),parity=(0,1,0))
            second = reconstruct_from_segment_nomes(table,inverse.segment_nomes,parity=(0,1,0))
            self.assertLess(abs(first.value-second.value),mp.mpf("1e-48"))
            self.assertEqual(first.value,first.prefactor*first.reduced_value)
        self.assertEqual(effective_plumbing_parameters((1,)),(16,))
        self.assertEqual(effective_plumbing_parameters((1,1,1)),(4,1,4))

    def test_ns_cap_product_coefficients(self):
        with mp.workdps(80):
            q = mp.mpf("1e-8")
            expected = 1+mp.mpf(11)/4*q*q+mp.mpf(93)/32*q**4
            self.assertLess(abs(ns_pillow_product(q)-expected),mp.mpf("1e-46"))
        self.assertEqual(ns_pillow_product(0),1)

    def test_full_prefactor_against_independent_plane_c_series_near_ope(self):
        with mp.workdps(80):
            for points in (4,5,6):
                data = fixture(points)
                table = compute(data,order=3,dps=70)
                p = tuple(mp.mpf("1e-9") for _ in range(points-3))
                z,mobiles = coordinates_from_segment_nomes(p)
                d,h = tuple(map(mp.mpf,data["external"])),tuple(map(mp.mpf,data["internal"]))
                xs = (z,) if not mobiles else (z/mobiles[0],)+tuple(a/b for a,b in zip(mobiles,mobiles[1:]))+(mobiles[-1],)
                leading = z**(h[0]-d[0]-d[1])*mp.fprod(t**(h[j+1]-h[j]-d[j+2]) for j,t in enumerate(mobiles))
                for eps in ((0,)*(points-3),(1,)*(points-3)):
                    plane = mp.mpf(0)
                    for row in data["coefficients"]:
                        key = row["twice_levels"]
                        if sum(key)>6 or tuple(n%2 for n in key) != eps:
                            continue
                        val = row["c_recursion_plane"]
                        plane += mp.mpc(val["real"],val["imag"])*mp.fprod(x**(mp.mpf(n)/2) for x,n in zip(xs,key))
                    full = reconstruct_from_segment_nomes(table,p,parity=eps).value/leading
                    # This is a small-coordinate truncation check, not exact
                    # equality of differently resummed finite-order functions.
                    self.assertLess(abs(full-plane),mp.mpf("1e-27"))


if __name__ == "__main__":
    unittest.main()
