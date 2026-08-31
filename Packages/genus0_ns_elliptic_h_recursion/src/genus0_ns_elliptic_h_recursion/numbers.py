"""Small input validators; no binary64 conversion in the numerical path."""

import mpmath as mp


def number(value, name="value"):
    result = mp.mpmathify(value)
    if not mp.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return mp.re(result) if mp.im(result) == 0 else result


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def precision(dps):
    nonnegative_integer(dps, "dps")
    if dps < 20:
        raise ValueError("dps must be at least 20")
    return dps


def twice_degree(value, maximum):
    if isinstance(value, bool):
        raise ValueError("degree must be a nonnegative integer or half-integer")
    doubled = 2*number(value, "degree")
    if mp.im(doubled) or doubled < 0 or doubled != int(doubled) or doubled > 2*maximum:
        raise ValueError(f"degree must be a half-integer between 0 and {maximum}")
    return int(doubled)


def parity_tuple(parity, edge_count):
    if parity is None:
        return (0,)*edge_count
    if len(parity) != edge_count or any(type(v) is not int or v not in (0,1) for v in parity):
        raise ValueError(f"parity must contain exactly {edge_count} integer bits")
    return tuple(parity)


def central_charge_from_b(b):
    b = number(b, "b")
    if not b:
        raise ValueError("b must be nonzero")
    return mp.mpf("1.5")+3*(b+1/b)**2


def b_from_c(c):
    q = mp.sqrt((number(c, "central_charge")-mp.mpf("1.5"))/3)
    return (q+mp.sqrt(q*q-4))/2
