"""Focused checks for the adaptive NS-block precision policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


FIT_DIR=Path(__file__).resolve().parent/"heterotic_so23_1to3_vvvv_fit_bundle"
sys.path.insert(0,str(FIT_DIR))

import heterotic_so23_1to3 as reference  # noqa: E402
import heterotic_so23_1to3_fast as stable  # noqa: E402


ENERGIES=(0.10j,0.12j,0.14j,0.36j)


def _weights(module):
    return tuple(module.h_of_p(value) for value in ENERGIES)


@pytest.mark.parametrize("momentum,reason",[(0.12,"low_p"),(1.0,"fast_exception")])
def test_adaptive_selector_uses_reference_on_dangerous_nodes(momentum,reason):
    h1,h2,h3,h4=_weights(stable)
    diagnostics={}
    _,_,backend=stable._select_block_pair(
        h4=h4,h3=h3,h2=h2,h1=h1,
        h_internal=stable.h_of_p(momentum),p=momentum,max_level2=21,
        reference_p_max=0.18,cancellation_limit=1.0e7,
        diagnostics=diagnostics,
    )
    assert backend == "reference"
    assert diagnostics[f"block_backend_reason_{reason}"] == 1


@pytest.mark.parametrize("star3,star2",[(False,False),(False,True),(True,False),(True,True)])
def test_validated_fast_coefficients_match_reference_through_q10(star3,star2):
    momentum=0.30
    h1f,h2f,h3f,h4f=_weights(stable)
    h1r,h2r,h3r,h4r=_weights(reference)
    fast=stable.FastNSBlockComputer(
        h4f,h3f,h2f,h1f,star3,star2,max_level2=21
    )
    exact=reference.NSBlockComputer(
        h4r,h3r,h2r,h1r,star3,star2,max_level2=21
    )
    hf=stable.h_of_p(momentum)
    hr=reference.h_of_p(momentum)
    for level2 in range(1,22):
        observed=complex(fast.coefficient(level2,hf))
        expected=complex(exact.coefficient(level2,hr))
        relative=abs(observed-expected)/max(abs(expected),1.0e-300)
        assert relative < 5.0e-10


@pytest.mark.parametrize("star3,star2",[(False,False),(True,True)])
@pytest.mark.parametrize("parity",["e","o"])
@pytest.mark.parametrize("derivative",[False,True])
def test_precomputed_direct_series_matches_reference(star3,star2,parity,derivative):
    h1,h2,h3,h4=_weights(reference)
    momentum=0.12
    h_internal=reference.h_of_p(momentum)
    block=reference.NSBlockComputer(
        h4,h3,h2,h1,star3,star2,max_level2=13
    )
    series=stable._build_direct_block_series(block,h_internal,6)
    z=np.asarray([0.025+0.010j,0.041-0.017j,0.060+0.003j])
    observed=stable._evaluate_direct_series(
        series,z,np.log(z),parity,derivative=derivative
    )
    expected=reference._direct_block(
        block,h_internal,z,parity,6,derivative=derivative
    )
    np.testing.assert_allclose(observed,expected,rtol=2.0e-14,atol=2.0e-14)


@pytest.mark.parametrize("momentum",[0.006,0.08,0.18,1.0])
def test_fast_structure_constants_match_reference(momentum):
    first=0.11+0.13j
    second=0.17+0.16j
    observed=(
        stable.c_even(first,second,momentum),
        stable.c_odd(first,second,momentum),
    )
    expected=(
        complex(reference.c_even(first,second,momentum)),
        complex(reference.c_odd(first,second,momentum)),
    )
    for value,target in zip(observed,expected):
        relative=abs(value-target)/max(abs(target),1.0e-300)
        assert relative < 1.0e-11
