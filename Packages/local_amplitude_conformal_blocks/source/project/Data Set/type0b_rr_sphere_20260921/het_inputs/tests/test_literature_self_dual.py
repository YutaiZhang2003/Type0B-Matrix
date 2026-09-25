from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Codes"))

from audit_literature_self_dual import run
from literature_self_dual_blocks import (
    ExtrapolatedLiteratureBlocks, analytic_antiholomorphic, duality_extrapolate,
)


def test_generic_b_extrapolation_and_independent_b1_references():
    report = run()
    assert report["status"] == "passed", report["rows"]


def test_analytic_antiholomorphic_keeps_physical_momenta():
    p = (.02+.22j,.03+.23j,.04+.24j,.09+.69j)
    block = ExtrapolatedLiteratureBlocks("mixed_r", p, .71, 4)
    dual = ExtrapolatedLiteratureBlocks("mixed_r", tuple(x.conjugate() for x in p), .71, 4)
    z, e, k, sl, sr = .5+.1j, (1,0,0,0), 1, -1, 1
    actual = analytic_antiholomorphic(dual)(z,e,k,sl,sr)
    naive = block.value(z,e,k,sl,sr).conjugate()
    assert abs(actual-naive) > .01*abs(actual)


def test_endpoint_estimator_cancels_linear_Q_minus_2_and_rejects_nonfinite():
    eps = .01
    f = lambda b: np.array([2+3j, -4j])+np.array([7-2j, 6])*(b-1)**2/b
    assert duality_extrapolate(f(1+eps),f(1+2*eps),eps) == pytest.approx(f(1), abs=1e-14)
    with pytest.raises(ValueError):
        duality_extrapolate([np.nan], [1], eps)
    with pytest.raises(ValueError):
        duality_extrapolate([1], [1,2], eps)
