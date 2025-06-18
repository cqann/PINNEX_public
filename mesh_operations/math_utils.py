import math

def uvc_to_cobi(a: float, r: float, m: float, v: int):
    """Convert UVC → CoBi (Eqs. 51-54)."""
    tv = v
    tm = 1 - m
    ab = a
    if v == 0:
        if abs(r) > math.pi/2:
            rt = 2/3 + 2/(3*math.pi) * math.atan2(math.cos(r), math.sin(r))
        else:
            rt = 2/3 + 1/(3*math.pi) * math.atan2(math.cos(r), math.sin(r))
    elif v == 1:
        rt = 1/3 + 2/(3*math.pi) * r
    else:
        raise ValueError("v must be 0 or 1")
    return ab, rt, tm, tv
