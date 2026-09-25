"""All-NS c-recursion with arbitrary external stars and a joint nome table.

The saved Type0B coefficients are human-note fixed-parity three-forms.
The polynomial-Ward conversion is explicit and independently PBW-testable.
No h-recursion, inverse Gram production fallback, or momentum conjugation.
"""
from functools import lru_cache
from itertools import product
from copy import copy

import mpmath as mp

from spin23_fivepoint_type0b_reference import layers


def bits(values, length):
    values = tuple(values)
    if len(values) != length or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError(f"{length} binary integer entries required")
    return values


def parity_path(stars, sectors):
    d, a = bits(stars, 5), bits(sectors, 3)
    r = (a[0] ^ d[0] ^ d[1], a[0] ^ a[1] ^ d[0] ^ d[1] ^ d[2])
    if r[1] ^ d[3] ^ d[4] != a[2]:
        raise ValueError("three-form sectors do not match the external parity")
    return r


def human_to_polynomial_phase(stars, sectors):
    """Product (-1)^(a*parity_of_zero_slot) over the three ordered vertices."""
    r = parity_path(stars, sectors)
    return (-1)**(sectors[0]*stars[0]+sectors[1]*r[0]+sectors[2]*r[1])


def so23_component_phase(stars, sectors):
    """Existing SO(23) physical-component dictionary, from polynomial forms.

    This is a basis conversion, not the separate time-fermion Wick sign.
    Keep both visible in the amplitude ledger.
    """
    r = parity_path(stars, sectors)
    exponent = stars[1]*(1-r[0])+stars[2]*(1-r[1])+stars[3]*(1-stars[4])
    return human_to_polynomial_phase(stars, sectors)*(-1)**exponent


@lru_cache(maxsize=1024)
def frame_at(x, y, precision):
    return layers().qw.QWFrame.from_plane(x, y, working_precision=precision)


class NSFivePointCBlock:
    def __init__(self, *, external_weights, internal_weights, stars, sectors,
                 maximum_twice_levels=(4, 4), precision=40, central_charge=13.5,
                 maximum_total_twice_level=None, coefficient_store=None):
        self.stars, self.sectors = bits(stars, 5), bits(sectors, 3)
        self.parities = parity_path(self.stars, self.sectors)
        self.maximum = tuple(maximum_twice_levels)
        if len(self.maximum) != 2 or any(type(v) is not int or v < 0 for v in self.maximum):
            raise ValueError("two nonnegative integer twice-level cutoffs required")
        self.maximum_total = sum(self.maximum) if maximum_total_twice_level is None else maximum_total_twice_level
        if type(self.maximum_total) is not int or not 0 <= self.maximum_total <= sum(self.maximum):
            raise ValueError("nonnegative total twice-level within the rectangle required")
        self._saved_table = None
        if type(precision) is not int or precision < 30:
            raise ValueError("precision must be an integer >=30")
        if len(external_weights) != 5 or len(internal_weights) != 2:
            raise ValueError("five external and two internal weights required")
        self.precision = precision
        with mp.workdps(precision):
            self.external = tuple(map(mp.mpc, external_weights))
            self.internal = tuple(map(mp.mpc, internal_weights))
            self.c = mp.mpc(central_charge)
            if not all(mp.isfinite(v) for v in (*self.external, *self.internal, self.c)):
                raise ValueError("finite conformal data required")
            self.effective = tuple(h+mp.mpf(d)/2 for h, d in zip(self.external, self.stars))
            self.leading = tuple(self.internal[i]-sum(self.effective[:i+2]) for i in (0, 1))
            self.raw = layers().CompactCBlock(central_charge=self.c,
                external_weights=self.external, internal_weights=self.internal,
                external_descendants=self.stars, vertex_sectors=self.sectors,
                working_precision=precision, pole_tolerance=1e-28,
                coefficient_store=coefficient_store)

    def phase(self, convention):
        if convention == "human":
            return 1
        if convention == "polynomial":
            return human_to_polynomial_phase(self.stars, self.sectors)
        if convention == "so23_component":
            return so23_component_phase(self.stars, self.sectors)
        if convention == "bry_coordinate_field":
            return (-1)**self.stars[-1]
        raise ValueError("unknown trilinear/component convention")

    def _cut(self, maximum):
        cut = self.maximum if maximum is None else tuple(maximum)
        if len(cut) != 2 or any(type(v) is not int or not 0 <= v <= m for v, m in zip(cut, self.maximum)):
            raise ValueError("requested coefficients exceed the compiled rectangle")
        return cut

    def coefficients(self, *, convention="human", maximum=None):
        cut = self._cut(maximum)
        keys = tuple(k for k in product(*(range(p, m+1, 2) for p, m in zip(self.parities, cut)))
            if sum(k) <= self.maximum_total)
        with mp.workdps(self.precision):
            self.raw._prepare(keys)
            return {k: self.phase(convention)*self.raw.final_coefficients[k] for k in keys}

    def table(self, maximum=None):
        cut = self._cut(maximum)
        if self._saved_table is not None:
            return {k: v for k, v in self._saved_table.items()
                if sum(k) <= self.maximum_total and all(a <= b for a, b in zip(k, cut))}
        with mp.workdps(self.precision):
            return self.raw.pillow_coefficients(cut, max_total_twice_level=self.maximum_total,
                effective_external_weights=self.effective)

    def prefix(self, physical_total_level):
        """A matched prefix of the SAME bank; half-integer parity is not reset."""
        if type(physical_total_level) is not int or not 0 <= 2*physical_total_level <= self.maximum_total:
            raise ValueError("physical total-level prefix exceeds available bank")
        result = copy(self)
        result._saved_table = self.table()
        result.maximum_total = 2*physical_total_level
        return result

    def value(self, x, y, *, convention="bry_coordinate_field", antiholomorphic=False,
              resummation="joint_elliptic", maximum=None):
        x, y = complex(x), complex(y)
        if any(not 0 < abs(z) < 1 for z in (x, y)):
            raise ValueError("noncolliding open sewing bidisk required")
        cut = self._cut(maximum)
        with mp.workdps(self.precision):
            if resummation == "joint_elliptic":
                value = self.raw.qw_block_value(frame_at(x, y, self.precision), cut,
                    max_total_twice_level=self.maximum_total,
                    effective_external_weights=self.effective, antiholomorphic=antiholomorphic)
            elif resummation == "sewing":
                logs = tuple(mp.log(z) for z in (x, y))
                if antiholomorphic:
                    logs = tuple(mp.conj(z) for z in logs)
                value = mp.exp(sum(a*l for a, l in zip(self.leading, logs))) * mp.fsum(
                    v*mp.exp((i*logs[0]+j*logs[1])/2)
                    for (i, j), v in self.coefficients(maximum=cut).items())
            else:
                raise ValueError("resummation must be joint_elliptic or sewing")
            return self.phase(convention)*value
