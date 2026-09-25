"""Finite-dimensional checks for the FULL coupled quartic Legendre transform.

No continuum, scattering, or quantum-domain assertion is tested here.
Run: /Users/sam/miniconda3/bin/python mqm_full_boundary_legendre_checks.py
"""
import json
from pathlib import Path
import numpy as np


def bisect_decreasing_rhs(rhs, upper):
    if upper == 0:
        return 0.0
    lo, hi = 0.0, upper
    for _ in range(180):
        mid = (lo + hi) / 2
        if mid > rhs(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def legendre(G, A, c, p):
    Ginvp = np.linalg.solve(G, p)
    B = A @ np.linalg.solve(G, A.T)
    y = A @ Ginvp
    ident = np.eye(A.shape[0])
    def boundary_velocity(sigma):
        return np.linalg.solve(ident + sigma * B, y)
    sigma = bisect_decreasing_rhs(
        lambda s: 4 * c * np.dot(boundary_velocity(s), boundary_velocity(s)),
        4 * c * np.dot(y, y),
    )
    w = boundary_velocity(sigma)
    v = np.linalg.solve(G, p - sigma * A.T @ w)
    T = 0.5 * v @ G @ v + 3 * c * np.dot(w, w) ** 2
    aux = 0.5 * p @ np.linalg.solve(G + sigma * A.T @ A, p) + sigma**2 / (16 * c)
    return {"T": T, "sigma": sigma, "v": v, "w": w, "aux": aux,
            "B": B, "y": y}


def boundary_legendre(M, c, k):
    ident = np.eye(len(k))
    def w_of(s):
        return np.linalg.solve(M + s * ident, k)
    # Since |(M+sI)^-1 k| <= |k|/s, the root lies below this bound.
    hi = (4 * c * np.dot(k, k)) ** (1 / 3)
    sigma = bisect_decreasing_rhs(lambda s: 4*c*np.dot(w_of(s), w_of(s)), hi)
    w = w_of(sigma)
    return 0.5*w@M@w + 3*c*np.dot(w,w)**2


def main():
    rng = np.random.default_rng(20260907)
    errors = {"stationarity": 0.0, "Av_equals_w": 0.0, "fenchel": 0.0,
              "auxiliary": 0.0, "schur": 0.0, "g_scaling": 0.0}
    min_vertical = float("inf")
    for sample in range(150):
        raw = rng.normal(size=(5, 5))
        G = raw.T@raw + 0.5*np.eye(5)
        A = rng.normal(size=(3, 5))
        if sample % 3 == 0:
            A[2] = 2*A[0]  # An exactly rank-deficient boundary map.
        if sample % 17 == 0:
            A[:] = 0
        p = rng.normal(size=5)
        c = float(np.exp(rng.uniform(-1, 1)))
        out = legendre(G, A, c, p)
        v, w, T = out["v"], out["w"], out["T"]
        L = 0.5*v@G@v + c*np.dot(A@v, A@v)**2
        scale = 1 + np.linalg.norm(p) + abs(T)
        errors["stationarity"] = max(errors["stationarity"],
            np.linalg.norm(p-G@v-4*c*np.dot(A@v,A@v)*A.T@(A@v))/scale)
        errors["Av_equals_w"] = max(errors["Av_equals_w"], np.linalg.norm(A@v-w)/scale)
        errors["fenchel"] = max(errors["fenchel"], abs(p@v-L-T)/scale)
        errors["auxiliary"] = max(errors["auxiliary"], abs(T-out["aux"])/scale)
        if np.linalg.matrix_rank(A) == A.shape[0]:
            M = np.linalg.inv(out["B"])
            k = M@out["y"]
            Kvert = 0.5*(p@np.linalg.solve(G,p)-out["y"]@M@out["y"])
            min_vertical = min(min_vertical, float(Kvert))
            schur = Kvert + boundary_legendre(M,c,k)
            errors["schur"] = max(errors["schur"], abs(T-schur)/scale)
        for g in (0.13, 0.7, 2.4):
            scaled = legendre(G/g**2, A, c/g**2, p/g**2)["T"]
            errors["g_scaling"] = max(errors["g_scaling"],abs(g**2*scaled-T)/scale)

    # Rank changes do not require a pseudoinverse in the definition.
    rank_rows = []
    G, p, c = np.diag([1.,2.,3.]), np.array([.7,1.1,-.2]), .6
    Tzero = legendre(G,np.array([[1.,0.,0.],[0.,0.,0.]]),c,p)["T"]
    for t in (1.,.1,.01,.001,0.):
        A = np.array([[1.,0.,0.],[0.,t,0.]])
        rank_rows.append({"t": t, "T": float(legendre(G,A,c,p)["T"]),
                          "difference_from_t0": float(legendre(G,A,c,p)["T"]-Tzero)})

    k, c = np.array([.7,-1.2]), .8
    T0 = 3*np.linalg.norm(k)**(4/3)/(4**(4/3)*c**(1/3))
    massless_rows = []
    for eps in (1.,.1,.01,.0001,.000001,.00000001):
        T = boundary_legendre(eps*np.diag([1.,3.]),c,k)
        massless_rows.append({"epsilon":eps,"T":float(T),"T0_minus_T":float(T0-T)})
    assert all(r["T0_minus_T"] >= -1e-12 for r in massless_rows)
    assert massless_rows[-1]["T0_minus_T"] < 1e-7

    # Equal quadratic boundary kinetic energy is not enough in anisotropic M.
    M = np.diag([1.,2.])
    k1, k2 = np.array([1.,0.]), np.array([0.,np.sqrt(2.)])
    anisotropy = {
        "quadratic_energies": [float(.5*k@np.linalg.solve(M,k)) for k in (k1,k2)],
        "quartic_coefficients_divided_by_minus_c": [float(np.linalg.norm(np.linalg.solve(M,k))**4) for k in (k1,k2)],
        "exact_T_at_c_1": [float(boundary_legendre(M,1.,k)) for k in (k1,k2)],
    }
    assert abs(anisotropy["exact_T_at_c_1"][0]-anisotropy["exact_T_at_c_1"][1]) > .01
    assert max(errors.values()) < 3e-12, errors
    assert min_vertical >= -1e-12
    results = {"random_cases":150,"max_normalized_errors":errors,
               "minimum_vertical_quadratic_energy":min_vertical,
               "rank_change":rank_rows,"anisotropic_zero_inertia_limit":massless_rows,
               "zero_inertia_limit":float(T0),"scalar_function_obstruction":anisotropy,
               "scope":"Classical finite-dimensional algebra only; no coupled quantum limit tested."}
    output = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_full_boundary_legendre_results.json')
    output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
