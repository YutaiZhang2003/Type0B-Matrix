"""Common coordinate sheets for the two mixed sphere channels.

Choose the lip in the original z plane BEFORE forming w=1-z. This helper
does not choose a new physical Ramond vertex convention or insert an
additional spin-frame phase into an existing coefficient tensor.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sphere_block_uniformization import CutSide, _cut_lip_argument


def branch_metadata(cut_side: CutSide = "upper") -> dict:
    if cut_side not in ("upper", "lower"):
        raise ValueError("cut_side must be upper or lower")
    return dict(
        schema="mixed-sphere-coordinate-sheet-v1",
        base_point=.5,
        domain="C minus (-infinity,0] and [1,infinity)",
        original_z_cut_side=cut_side,
        antiholomorphic_lip="opposite original z lip",
        crossed_coordinate="1-z, including the transported infinitesimal",
        square_root="positive for 0<z<1; continued with the coordinate",
        momenta_conjugated=False,
        external_spin_frame="existing ordered vertex phases retained; physical dictionary uncertified",
    )


def require_same_coordinate_branch(left_parameters, right_parameters):
    """Reject a comparison/refinement with different declared sheets.

    Two historical reports without metadata remain readable, but do not
    acquire a retroactive certification from this helper.
    """
    left = left_parameters.get("coordinate_branch")
    right = right_parameters.get("coordinate_branch")
    if left != right:
        raise ValueError("incompatible coordinate_branch: compare the same original-plane sheet")
    return left


@dataclass(frozen=True)
class SphereChannelCoordinates:
    z: complex | np.ndarray
    zbar: complex | np.ndarray
    local: complex | np.ndarray
    local_bar: complex | np.ndarray
    holomorphic_cut_side: CutSide
    antiholomorphic_cut_side: CutSide


def sphere_channel_coordinates(z, *, channel: Literal["direct", "crossed"],
                               zbar=None, cut_side: CutSide = "upper"):
    """Transport a common original-plane boundary value into either chart.

    Nonreal coordinates are retained exactly. On a real cut, holomorphic
    z and geometric zbar have opposite infinitesimals. The map 1-z reverses
    both. Independently supplied nonreal zbar is never conjugated.
    """
    branch_metadata(cut_side)  # Validate the declared original-plane lip.
    if channel not in ("direct", "crossed"):
        raise ValueError("channel must be direct or crossed")
    opposite = "lower" if cut_side == "upper" else "upper"
    original = np.asarray(z, dtype=complex)
    scalar = original.ndim == 0
    if np.any(~np.isfinite(original)) or np.any((original == 0) | (original == 1)):
        raise ValueError("coordinates must be finite and avoid 0 and 1")
    lifted = _cut_lip_argument(original, cut_side)
    if zbar is None:
        anti = lifted.conjugate()
    else:
        anti = np.asarray(zbar, dtype=complex)
        if anti.shape != original.shape or np.any(~np.isfinite(anti)) or np.any((anti == 0) | (anti == 1)):
            raise ValueError("zbar must match z's shape and avoid punctures")
        anti = _cut_lip_argument(anti, opposite)
    local, local_bar = (lifted, anti) if channel == "direct" else (1-lifted, 1-anti)
    hs, bs = (cut_side, opposite) if channel == "direct" else (opposite, cut_side)
    convert = (lambda x: complex(x.item())) if scalar else (lambda x: x)
    return SphereChannelCoordinates(*(convert(x) for x in (lifted, anti, local, local_bar)), hs, bs)
