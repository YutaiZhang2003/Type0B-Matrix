"""Stable manifest adapter for the Spin(23) heterotic 1-to-3 amplitude scan.

The public ``evaluate_point`` function follows the contract in
``spin23_highdim_scan.py``.  Numerical coefficients are evaluated in the new
bundle convention (A,B,C) and returned in the scan convention
``(M1,M2,M3)=(C,B,A)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import mpmath as mp
import numpy as np


FIT_DIR=Path(__file__).resolve().parent/"heterotic_so23_1to3_vvvv_fit_bundle"
sys.path.insert(0,str(FIT_DIR))

from heterotic_so23_1to3_fast import (  # noqa: E402
    pair_channel_ansatz,
    vector_amplitude_coefficients_regularized,
)
import ns_elliptic_conversion as elliptic_conversion


def _numerical_options(settings: Mapping[str,Any],diagnostics: dict[str,Any]) -> dict[str,Any]:
    series_parameter=elliptic_conversion.validate_representation(settings.get("series_parameter", "elliptic_nome"))
    angular=int(settings.get("z_angular_nodes",96))
    radial=int(settings.get("z_radial_nodes",48))
    q_order=int(settings.get("q_order",8))
    return {
        "series_parameter":series_parameter,
        "q_order":q_order,
        "p_nodes":settings.get("p_nodes",72),
        "p_max":float(settings.get("p_max",6.0)),
        "momentum_scheme":str(settings.get("momentum_scheme","threshold_weighted")),
        "momentum_threshold_options":settings.get("momentum_threshold_options"),
        "epsilon0":float(settings.get("ope_radius",0.02)),
        "epsilon1":float(settings.get("crossed_ope_radius",0.06)),
        "theta_orders":(
            max(12,angular//4),max(12,angular//4),angular,
        ),
        "radial_order":radial,
        "disk_total_order":int(settings.get("disk_total_order",q_order+8)),
        "lens_radial_order":int(settings.get("lens_radial_nodes",max(20,radial//2))),
        "lens_angular_order":int(settings.get("lens_angular_nodes",max(48,2*angular//3))),
        "lens_power":float(settings.get("lens_power",3.0)),
        "p_cut":float(settings.get("p_cut",0.03)),
        "reference_p_max":float(settings.get("reference_p_max",0.18)),
        "cancellation_limit":float(settings.get("cancellation_limit",1.0e7)),
        "diagnostics":diagnostics,
    }


def _old_channel_order(values) -> np.ndarray:
    a,b,c=np.asarray(values,dtype=complex)
    return np.array([c,b,a],dtype=complex)


def evaluate_point(
    omega1: complex,
    omega2: complex,
    omega3: complex,
    settings: Mapping[str,Any],
) -> dict[str,Any]:
    """Evaluate one manifest row with optional adjacent-order error estimation."""
    mp.mp.dps=int(settings.get("mp_dps",80))
    energies=[complex(omega1),complex(omega2),complex(omega3)]
    energies.append(sum(energies))

    diagnostics: dict[str,Any]={}
    options=_numerical_options(settings,diagnostics)
    options["structure_cache"]={}
    options["block_cache"]={}
    options["elliptic_cache"]={}
    options["direct_series_cache"]={}
    numerical=_old_channel_order(
        vector_amplitude_coefficients_regularized(energies,**options)
    )
    candidate=_old_channel_order(pair_channel_ansatz(energies))

    estimate_error=bool(settings.get("estimate_q_error",True))
    if estimate_error and options["q_order"] > 2:
        lower_diagnostics: dict[str,Any]={}
        lower_options=dict(options)
        lower_options["q_order"]-=1
        lower_options["diagnostics"]=lower_diagnostics
        lower=_old_channel_order(
            vector_amplitude_coefficients_regularized(energies,**lower_options)
        )
        estimated_error=np.abs(numerical-lower)
    else:
        lower_diagnostics={}
        estimated_error=np.full(3,np.nan)

    residual=np.abs(numerical-candidate)
    relative=residual/np.maximum(np.abs(candidate),np.finfo(float).tiny)
    result: dict[str,Any]={}
    for index,name in enumerate(("M1","M2","M3")):
        result[name]=numerical[index]
        result[f"candidate_{name}_re"]=candidate[index].real
        result[f"candidate_{name}_im"]=candidate[index].imag
        result[f"formula_abs_residual_{name}"]=float(residual[index])
        result[f"formula_rel_residual_{name}"]=float(relative[index])
        result[f"estimated_abs_error_{name}"]=float(estimated_error[index])
        result[f"estimated_rel_error_{name}"]=float(
            estimated_error[index]/max(abs(numerical[index]),np.finfo(float).tiny)
        )
        result[f"formula_residual_over_q_change_{name}"]=float(
            residual[index]/max(estimated_error[index],np.finfo(float).tiny)
        )

    result.update({f"block_{key}":value for key,value in diagnostics.items()})
    result["lower_order_block_diagnostics"]=json.dumps(
        lower_diagnostics,sort_keys=True,separators=(",",":")
    )
    result.update(elliptic_conversion.representation_metadata(options["series_parameter"],options["q_order"]))
    result["momentum_scheme"]=options["momentum_scheme"]
    result["momentum_quadrature"]=json.dumps(diagnostics.get("momentum_quadrature",{}),sort_keys=True)
    result["mp_dps"]=mp.mp.dps
    result["status"]="ok"
    return result
