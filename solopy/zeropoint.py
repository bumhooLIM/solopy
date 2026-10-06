"""
Zero-point helpers: a per-frame color term, and local (position-dependent) zero points.

The color term refers every star's zero point to solar color, so asteroids (roughly solar
colors) are not calibrated with the mean color of field stars (robustness review R3).
Local zero points follow extinction gradients across the 3.4 deg field and fixed detector
patterns that a single value per frame cannot (robustness review R2).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = ["SOLAR_BP_RP", "robust_std", "fit_color_term", "local_zero_points"]

SOLAR_BP_RP = 0.82  # Gaia DR3 BP-RP of the Sun (Casagrande & VandenBerg 2018)


def robust_std(values):
    """1.4826 * median absolute deviation (NaN for an empty input)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def _clip(values, nsigma=3.0, maxiters=5):
    """Boolean mask of the values kept by iterative median/MAD clipping."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    for _ in range(maxiters):
        if keep.sum() < 3:
            break
        med, s = np.median(v[keep]), robust_std(v[keep])
        if not np.isfinite(s) or s == 0:
            break
        new = keep & (np.abs(v - med) <= nsigma * s)
        if (new == keep).all():
            break
        keep = new
    return keep


def fit_color_term(zp_star, bp_rp, ref_color=SOLAR_BP_RP, nsigma=3.0, maxiters=5, min_stars=10):
    """
    Fit zp_star = zp_ref + slope * (bp_rp - ref_color) with iterative clipping.

    Returns (zp_ref, slope, n_used): the zero point at `ref_color`, the color slope
    [mag per mag of BP-RP], and the number of stars kept. With fewer than `min_stars`
    stars with a color, the slope is 0 and zp_ref is the clipped median.
    """
    z = np.asarray(zp_star, dtype=float)
    c = np.asarray(bp_rp, dtype=float)
    ok = np.isfinite(z) & np.isfinite(c)
    if ok.sum() < min_stars:
        keep = _clip(z, nsigma, maxiters)
        return (float(np.median(z[keep])) if keep.any() else np.nan), 0.0, int(keep.sum())

    x, y = c[ok] - ref_color, z[ok]
    keep = np.ones(x.size, dtype=bool)
    coef = np.array([np.median(y), 0.0])
    for _ in range(maxiters):
        A = np.column_stack([np.ones(keep.sum()), x[keep]])
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        resid = y - (coef[0] + coef[1] * x)
        s = robust_std(resid[keep])
        if not np.isfinite(s) or s == 0:
            break
        new = np.abs(resid - np.median(resid[keep])) <= nsigma * s
        if (new == keep).all():
            break
        keep = new
    return float(coef[0]), float(coef[1]), int(keep.sum())


def local_zero_points(star_x, star_y, star_zp, target_x, target_y, radius=500.0, min_stars=10,
                      fallback_zp=np.nan, fallback_err=np.nan, nsigma=3.0):
    """
    Zero point at each target from the stars within `radius` pixels.

    Returns a DataFrame (one row per target) with
      zp_local          clipped median of the neighbouring stars' zero points
      zperr_local       its standard error, 1.2533 * robust_std / sqrt(n)
      zp_local_spread   robust scatter of those stars (large in cloudy or twilight frames)
      zp_local_n        number of stars used
      zp_local_fallback True when fewer than `min_stars` neighbours were available; the
                        target then gets `fallback_zp` / `fallback_err`.
    """
    sx, sy, sz = (np.asarray(a, dtype=float) for a in (star_x, star_y, star_zp))
    good = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
    sx, sy, sz = sx[good], sy[good], sz[good]
    tx, ty = np.atleast_1d(np.asarray(target_x, dtype=float)), np.atleast_1d(np.asarray(target_y, dtype=float))

    rows = []
    tree = cKDTree(np.column_stack([sx, sy])) if sz.size else None
    for xt, yt in zip(tx, ty):
        idx = tree.query_ball_point([xt, yt], radius) if tree is not None and np.isfinite(xt + yt) else []
        if len(idx) >= min_stars:
            vals = sz[idx]
            vals = vals[_clip(vals, nsigma)]
            spread = robust_std(vals)
            rows.append((float(np.median(vals)), 1.2533 * spread / np.sqrt(vals.size), spread, vals.size, False))
        else:
            rows.append((float(fallback_zp), float(fallback_err), np.nan, len(idx), True))
    return pd.DataFrame(rows, columns=["zp_local", "zperr_local", "zp_local_spread", "zp_local_n", "zp_local_fallback"])
