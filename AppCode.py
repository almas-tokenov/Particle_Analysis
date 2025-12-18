
import io
import os
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Optional KDE in analytics
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Required for ELLE()
try:
    from scipy.special import elliprd, elliprf
    SCIPY_SPECIAL_AVAILABLE = True
except Exception:
    SCIPY_SPECIAL_AVAILABLE = False


# =============================================================================
# Optimized Shape Analysis (refactored to avoid globals)
# =============================================================================

def ELLE(ak: float) -> float:
    """Estimate complete elliptic integral (same approach as your legacy code)."""
    if not SCIPY_SPECIAL_AVAILABLE:
        raise RuntimeError("scipy.special (elliprf/elliprd) is required for ELLE().")

    # clamp ak into [0, 1] for stability
    ak = float(np.clip(ak, 0.0, 1.0))
    pi2 = np.pi / 2.0
    s = 1.0  # sin(pi/2)
    cc = 0.0  # cos(pi/2)^2
    Q = (1.0 - s * ak) * (1.0 + s * ak)
    return s * (elliprf(cc, Q, 1.0) - ((s * ak) * (s * ak)) * elliprd(cc, Q, 1.0) / 3.0)


def OutlineCentroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(x)), float(np.mean(y))


def OutlineArea_optimized(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula (vectorized), closed by connecting last->first automatically."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_rolled = np.roll(x, -1)
    y_rolled = np.roll(y, -1)
    cross_sum = np.sum(x * y_rolled - x_rolled * y)
    return float(np.abs(cross_sum) / 2.0)


def OutlineCircumference_optimized(x: np.ndarray, y: np.ndarray) -> float:
    """Perimeter of closed polygon, vectorized."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def ComputeEllFourierCoef_optimized(
    x: np.ndarray, y: np.ndarray, n_harmonics: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Elliptic Fourier Descriptors (EFDs) for a closed contour.

    Returns Ax, Bx, Ay, By with DC terms in index 0.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_pts = len(x)

    Ax = np.zeros(n_harmonics + 1, dtype=np.float64)
    Bx = np.zeros(n_harmonics + 1, dtype=np.float64)
    Ay = np.zeros(n_harmonics + 1, dtype=np.float64)
    By = np.zeros(n_harmonics + 1, dtype=np.float64)

    if n_pts < 2:
        Ax[0] = x[0] if n_pts else 0.0
        Ay[0] = y[0] if n_pts else 0.0
        return Ax, Bx, Ay, By

    # segment diffs including closing segment (roll)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    dx = x_next - x
    dy = y_next - y
    dt = np.sqrt(dx * dx + dy * dy)

    # avoid division by zero segments
    rx = np.where(dt != 0.0, dx / dt, 0.0)
    ry = np.where(dt != 0.0, dy / dt, 0.0)

    t_curr = np.cumsum(dt)
    t_prev = np.concatenate(([0.0], t_curr[:-1]))
    T = float(t_curr[-1]) if len(t_curr) else 0.0

    # DC terms via stable cumulative loop (matches your optimized logic)
    Tsum = 0.0
    Xsum = 0.0
    Ysum = 0.0
    Ax0_int = 0.0
    Ay0_int = 0.0
    for i in range(n_pts):
        DT_i = float(dt[i])
        Tnew = Tsum + DT_i
        Rx_i = float(rx[i])
        Ry_i = float(ry[i])

        Xi_i = Xsum - Rx_i * Tsum
        Delta_i = Ysum - Ry_i * Tsum

        Ax0_int += 0.5 * Rx_i * (Tnew * Tnew - Tsum * Tsum) + Xi_i * DT_i
        Ay0_int += 0.5 * Ry_i * (Tnew * Tnew - Tsum * Tsum) + Delta_i * DT_i

        Tsum = Tnew
        Xsum += float(dx[i])
        Ysum += float(dy[i])

    if T > 0.0:
        Ax0 = float(x[0] + Ax0_int / T)
        Ay0 = float(y[0] + Ay0_int / T)
    else:
        Ax0 = float(x[0])
        Ay0 = float(y[0])

    Ax[0] = Ax0
    Ay[0] = Ay0

    if n_harmonics == 0 or T == 0.0:
        return Ax, Bx, Ay, By

    j = np.arange(1, n_harmonics + 1, dtype=np.float64)
    c1 = 2.0 * np.pi * j / T

    theta_curr = np.outer(t_curr, c1)
    theta_prev = np.outer(t_prev, c1)

    diff_cos = np.cos(theta_curr) - np.cos(theta_prev)
    diff_sin = np.sin(theta_curr) - np.sin(theta_prev)

    AxSUM = rx @ diff_cos
    BxSUM = rx @ diff_sin
    AySUM = ry @ diff_cos
    BySUM = ry @ diff_sin

    c2 = T / (2.0 * (np.pi ** 2) * (j ** 2))
    Ax[1:] = AxSUM * c2
    Bx[1:] = BxSUM * c2
    Ay[1:] = AySUM * c2
    By[1:] = BySUM * c2

    return Ax, Bx, Ay, By


def FourierCoefNormalization(
    Ax: np.ndarray,
    Bx: np.ndarray,
    Ay: np.ndarray,
    By: np.ndarray,
    flag_scale: int,
    flag_rotation: int,
    flag_start: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Normalize Fourier coefficients (scale, rotation, start invariance).
    Location invariance is handled by centering (x-xmid, y-ymid) before EFD.
    """
    Ax = Ax.copy()
    Bx = Bx.copy()
    Ay = Ay.copy()
    By = By.copy()

    Ax0, Ay0 = Ax[0], Ay[0]
    Ax1, Bx1, Ay1, By1 = Ax[1], Bx[1], Ay[1], By[1]

    denom = Ax1 * Ax1 + Ay1 * Ay1 - Bx1 * Bx1 - By1 * By1
    theta = 0.5 * np.arctan2(2.0 * (Ax1 * Bx1 + Ay1 * By1), denom)

    Astar = Ax1 * np.cos(theta) + Bx1 * np.sin(theta)
    Cstar = Ay1 * np.cos(theta) + By1 * np.sin(theta)
    psi = np.arctan2(Cstar, Astar)

    Estar = float(np.sqrt(Astar * Astar + Cstar * Cstar))
    if Estar == 0.0:
        Estar = 1.0
    if flag_scale == 0:
        Estar = 1.0
    if flag_rotation == 0:
        psi = 0.0
    if flag_start == 0:
        theta = 0.0

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    Ax0N = (cos_psi * Ax0 + sin_psi * Ay0) / Estar
    Ay0N = (-sin_psi * Ax0 + cos_psi * Ay0) / Estar

    n_harmonics = len(Ax) - 1
    for i in range(1, n_harmonics + 1):
        cos_t = np.cos(float(i) * theta)
        sin_t = np.sin(float(i) * theta)

        c1 = Ax[i] * cos_psi + Ay[i] * sin_psi
        c2 = Bx[i] * cos_psi + By[i] * sin_psi
        c3 = Ay[i] * cos_psi - Ax[i] * sin_psi
        c4 = By[i] * cos_psi - Bx[i] * sin_psi

        Ax[i] = (c1 * cos_t + c2 * sin_t) / Estar
        Bx[i] = (c2 * cos_t - c1 * sin_t) / Estar
        Ay[i] = (c3 * cos_t + c4 * sin_t) / Estar
        By[i] = (c4 * cos_t - c3 * sin_t) / Estar

    Ax[0] = Ax0N
    Ay[0] = Ay0N
    Bx[0] = 0.0
    By[0] = 0.0

    return Ax, Bx, Ay, By, Estar, float(psi), float(theta)


def CoefNormalization(
    x: np.ndarray,
    y: np.ndarray,
    xmid: float,
    ymid: float,
    flag_location: int,
    flag_scale: int,
    flag_rotation: int,
    flag_start: int,
    n_harmonics: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compute EFD then apply normalization; no global variables."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if flag_location == 1:
        x0 = x - xmid
        y0 = y - ymid
    else:
        x0 = x
        y0 = y

    Ax, Bx, Ay, By = ComputeEllFourierCoef_optimized(x0, y0, n_harmonics)
    Ax, Bx, Ay, By, scale1, rotate_angle, start_angle = FourierCoefNormalization(
        Ax, Bx, Ay, By,
        flag_scale=flag_scale,
        flag_rotation=flag_rotation,
        flag_start=flag_start,
    )
    return Ax, Bx, Ay, By, scale1, rotate_angle, start_angle


def ShapeIndex_optimized(
    Ax: np.ndarray,
    Bx: np.ndarray,
    Ay: np.ndarray,
    By: np.ndarray,
    flag_rotation: int,
    psi: float,
    xmid: float,
    ymid: float,
    scale1: float,
    x: np.ndarray,
    y: np.ndarray,
    n0_sum: int,
    large_number: float,
):
    """
    Returns: ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk
    """
    Bk1 = np.zeros(64, dtype=np.float64)
    Ask = np.zeros(max(64, n0_sum + 1), dtype=np.float64)
    Bsk = np.zeros(max(64, n0_sum + 1), dtype=np.float64)
    Bk2 = np.zeros(n0_sum + 1, dtype=np.float64)

    Ax1, Bx1, Ay1, By1 = Ax[1], Bx[1], Ay[1], By[1]
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))

    if flag_rotation == 0:
        denom = Ax1**2 + Ay1**2 - Bx1**2 - By1**2
        theta = 0.5 * np.arctan2(2.0 * (Ax1 * Bx1 + Ay1 * By1), denom)
        Astar = Ax1 * np.cos(theta) + Bx1 * np.sin(theta)
        Cstar = Ay1 * np.cos(theta) + By1 * np.sin(theta)
        psi = float(np.arctan2(Cstar, Astar))
        cos_psi = float(np.cos(psi))
        sin_psi = float(np.sin(psi))

        c1 = Ax1 * cos_psi + Ay1 * sin_psi
        c2 = Bx1 * cos_psi + By1 * sin_psi
        c3 = Ay1 * cos_psi - Ax1 * sin_psi
        c4 = By1 * cos_psi - Bx1 * sin_psi
        Ax1, Bx1, Ay1, By1 = c1, c2, c3, c4

    det = float(np.abs(Bx1 * Ay1 - Ax1 * By1))
    denom_ae = float(np.sqrt(Ay1**2 + By1**2))
    denom_be = float(np.sqrt(Ax1**2 + Bx1**2))
    ae = det / denom_ae if denom_ae != 0 else np.nan
    be = det / denom_be if denom_be != 0 else np.nan
    k = be / ae if (np.isfinite(ae) and ae != 0) else np.nan

    # Unevenness coefficient Uc = Sr/rc
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pts = len(x)

    if not np.isfinite(ae) or not np.isfinite(be) or ae <= 0 or be <= 0 or scale1 == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, Bk1, Bk2, Ask[:64], Bsk[:64])

    Xs = x - xmid
    Ys = y - ymid
    xp = (Xs * cos_psi + Ys * sin_psi) / scale1
    yp = (-Xs * sin_psi + Ys * cos_psi) / scale1

    rd = np.sqrt(xp * xp + yp * yp)
    th = np.arctan2(yp, xp)

    ae_sq = ae * ae
    be_sq = be * be
    cos_th_sq = np.cos(th) ** 2

    re_denom = np.sqrt(np.maximum(ae_sq - (ae_sq - be_sq) * cos_th_sq, 0.0))
    re = np.divide(ae * be, re_denom, out=np.full_like(re_denom, np.nan), where=re_denom != 0)

    Sr = float(np.sum((rd - re) ** 2) / float(n_pts))

    xk = 1.0 - (be_sq / ae_sq)
    xk = float(np.clip(xk, 0.0, 1.0))
    Le = float(4.0 * ae * ELLE(np.sqrt(xk)))
    rc = float(Le / (2.0 * np.pi)) if Le != 0 else np.nan
    Uc = float(np.sqrt(Sr) / rc) if (np.isfinite(rc) and rc != 0) else np.nan

    # Sharpness S1, S2 (vectorized version from your optimized implementation)
    xp1 = np.roll(xp, -1)
    yp1 = np.roll(yp, -1)
    dx = xp1 - xp
    dy = yp1 - yp

    di = np.full_like(dx, 1000.0, dtype=np.float64)
    mask = np.abs(dx) > 1e-30
    np.divide(dy, dx, out=di, where=mask)

    jj = np.arange(36, dtype=np.float64)
    theta_jj = (2.0 * np.pi / 36.0) * jj
    cos_jj = np.cos(theta_jj)
    sin_jj = np.sin(theta_jj)

    di2 = di[:, None]
    numerator = di2 * cos_jj - sin_jj
    denominator = cos_jj + di2 * sin_jj
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    aver_sum = np.sum(ratio, axis=1)
    aver = np.arctan(aver_sum / 36.0)

    # S1 coefficients for j=1..64
    j1 = np.arange(1, 65, dtype=np.float64)
    i1 = np.arange(1, n_pts + 1, dtype=np.float64)
    arg1 = (2.0 * np.pi / n_pts) * i1[:, None] * j1[None, :]
    say_vec = np.sum(aver[:, None] * np.cos(arg1), axis=0) / float(n_pts)
    sby_vec = np.sum(aver[:, None] * np.sin(arg1), axis=0) / float(n_pts)

    sint1 = np.sqrt(say_vec * say_vec + sby_vec * sby_vec)
    SUM_Bk1 = float(np.sum(sint1))
    S1 = float(1.0 / SUM_Bk1) if SUM_Bk1 != 0 else np.nan
    Bk1[:] = sint1
    Ask[:64] = say_vec
    Bsk[:64] = sby_vec

    # S2 coefficients for j=1..n0_sum
    j2 = np.arange(1, n0_sum + 1, dtype=np.float64)
    arg2 = (2.0 * np.pi / n_pts) * i1[:, None] * j2[None, :]
    say2 = np.sum(aver[:, None] * np.cos(arg2), axis=0) / float(n_pts)
    sby2 = np.sum(aver[:, None] * np.sin(arg2), axis=0) / float(n_pts)

    say1 = float(say2[0]) if len(say2) else np.nan
    sby1 = float(sby2[0]) if len(sby2) else np.nan

    SUM_Bk2 = float(np.sqrt(2.0))
    Bk2[1] = float(np.sqrt(2.0))

    if n0_sum >= 2 and np.isfinite(say1) and np.isfinite(sby1) and say1 != 0 and sby1 != 0:
        sint2 = np.sqrt((say2[1:] / say1) ** 2 + (sby2[1:] / sby1) ** 2)
        sint2 = np.where(sint2 > large_number, 0.0, sint2)
        SUM_Bk2 += float(np.sum(sint2))
        Bk2[2:n0_sum + 1] = sint2

    S2 = float(1.0 / SUM_Bk2) if SUM_Bk2 != 0 else np.nan
    Ask[1:n0_sum + 1] = say2
    Bsk[1:n0_sum + 1] = sby2

    return ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask[:max(64, n0_sum + 1)], Bsk[:max(64, n0_sum + 1)]


def ShapeIndicesEF_optimized(
    Ax: np.ndarray,
    Bx: np.ndarray,
    Ay: np.ndarray,
    By: np.ndarray,
    no_sum: int,
    large_limit: float,
    small_limit: float,
    xmid: float,
    ymid: float,
    flag_scale: int,
    scale1: float,
    psi: float,
    start_angle: float,  # kept for API compatibility
    x: np.ndarray,
    y: np.ndarray,
    n_harmonics: int,
    flag_rotation: int,
):
    """
    Returns:
      ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P
    """
    # shape index (ellipse, roughness, sharpness)
    ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk = ShapeIndex_optimized(
        Ax, Bx, Ay, By,
        flag_rotation=flag_rotation,
        psi=psi,
        xmid=xmid,
        ymid=ymid,
        scale1=1.0 if flag_scale == 0 else scale1,
        x=x,
        y=y,
        n0_sum=no_sum,
        large_number=large_limit,
    )

    # asymmetricity + polygonality use EFD recomputed from (x,y) (kept consistent with your code)
    Ax2, Bx2, Ay2, By2 = ComputeEllFourierCoef_optimized(x, y, n_harmonics)

    # Ass2
    no_ass = min(n_harmonics, no_sum)
    Ax_s = Ax2[1:no_ass + 1]
    Bx_s = Bx2[1:no_ass + 1]
    Ay_s = Ay2[1:no_ass + 1]
    By_s = By2[1:no_ass + 1]

    abs_Ax = np.abs(Ax_s)
    abs_Bx = np.abs(Bx_s)
    abs_Ay = np.abs(Ay_s)
    abs_By = np.abs(By_s)

    cond_bx = (abs_Bx > small_limit) & (abs_Ax > small_limit)
    cond_ay = (abs_Ay > small_limit) & (abs_By > small_limit)

    ratio_bx = np.divide(abs_Bx, abs_Ax, out=np.zeros_like(abs_Bx), where=cond_bx)
    ratio_ay = np.divide(abs_Ay, abs_By, out=np.zeros_like(abs_Ay), where=cond_ay)

    SumBx = float(np.sum(ratio_bx))
    SumAy = float(np.sum(ratio_ay))
    Ass2 = float(np.sqrt(SumBx * SumAy)) if (SumBx >= 0 and SumAy >= 0) else np.nan

    # Ass1 (j = 2..min(15, n_harmonics))
    no_ass1 = min(15, n_harmonics)
    Ax_s1 = Ax2[2:no_ass1 + 1]
    Ay_s1 = Ay2[2:no_ass1 + 1]
    Bx_s1 = Bx2[2:no_ass1 + 1]
    By_s1 = By2[2:no_ass1 + 1]

    SumAx = float(np.sum(np.where(np.abs(Ax_s1) > small_limit, np.abs(Ax_s1), 0.0)))
    SumAy = float(np.sum(np.where(np.abs(Ay_s1) > small_limit, np.abs(Ay_s1), 0.0)))
    SumBx = float(np.sum(np.where(np.abs(Bx_s1) > small_limit, np.abs(Bx_s1), 0.0)))
    SumBy = float(np.sum(np.where(np.abs(By_s1) > small_limit, np.abs(By_s1), 0.0)))

    Ass1 = 0.0
    if SumAx > 0.0 and SumBy > 0.0:
        Ass1 = float(np.sqrt((SumBx / SumAx) * (SumAy / SumBy)))

    # polygonality P (kept as in your code)
    # find first max indices
    j = 1
    MaxAx = float(np.abs(Ax2[2])) if len(Ax2) > 2 else 0.0
    NoMaxAx1 = 2
    MaxBy = float(np.abs(By2[2])) if len(By2) > 2 else 0.0
    NoMaxBy1 = 2

    while j <= n_harmonics:
        if MaxAx < float(np.abs(Ax2[j])):
            MaxAx = float(np.abs(Ax2[j])); NoMaxAx1 = j
        if MaxBy < float(np.abs(By2[j])):
            MaxBy = float(np.abs(By2[j])); NoMaxBy1 = j
        j += 1

    j = 1
    MaxAx = float(np.abs(Ax2[2])) if len(Ax2) > 2 else 0.0
    NoMaxAx = 2
    MaxBy = float(np.abs(By2[2])) if len(By2) > 2 else 0.0
    NoMaxBy = 2

    while j <= n_harmonics:
        if j != NoMaxAx1 and MaxAx < float(np.abs(Ax2[j])):
            MaxAx = float(np.abs(Ax2[j])); NoMaxAx = j
        if j != NoMaxBy1 and MaxBy < float(np.abs(By2[j])):
            MaxBy = float(np.abs(By2[j])); NoMaxBy = j
        j += 1

    Px = NoMaxAx + 1
    Py = NoMaxBy + 1
    P = float(np.sqrt(Px * Py))

    return ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P


def ComputeAngularity_optimized(
    Ax: np.ndarray, Bx: np.ndarray, Ay: np.ndarray, By: np.ndarray, n_harmonics: int, w: int = 360
) -> float:
    """Angularity index using vectorized derivative computation."""
    TwoPi = 2.0 * np.pi
    angles = np.linspace(0.0, TwoPi, w, endpoint=False)

    n = np.arange(1, n_harmonics + 1, dtype=np.float64)
    n_u = np.outer(angles, n)

    sin_nu = np.sin(n_u)
    cos_nu = np.cos(n_u)

    nAx = n * Ax[1:n_harmonics + 1]
    nBx = n * Bx[1:n_harmonics + 1]
    nAy = n * Ay[1:n_harmonics + 1]
    nBy = n * By[1:n_harmonics + 1]

    x_deriv = np.sum((-nAx) * sin_nu + (nBx) * cos_nu, axis=1)
    y_deriv = np.sum((-nAy) * sin_nu + (nBy) * cos_nu, axis=1)

    hx = np.unwrap(np.arctan2(y_deriv, x_deriv))
    dh = np.abs(np.diff(hx, append=hx[0]))
    return float((1.0 / TwoPi) * np.sum(dh) - 1.0)


def ShapeFactors(
    x: np.ndarray, y: np.ndarray, area: Optional[float] = None, perimeter: Optional[float] = None
) -> Tuple[float, float, float, float, float]:
    """
    Elongation, bulkiness, surface, circularity, sphericity.
    (Kept consistent with your existing code, minor safety checks.)
    """
    Pi = np.pi
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if area is None:
        area = OutlineArea_optimized(x, y)
    if perimeter is None:
        perimeter = OutlineCircumference_optimized(x, y)

    if area <= 0 or perimeter <= 0 or not np.isfinite(area) or not np.isfinite(perimeter):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    Xm, Ym = float(np.mean(x)), float(np.mean(y))

    # rotation search (as in your original ShapeFactors)
    N_angle = 180
    delta_psi = Pi / (N_angle - 1)
    psi = 0.0
    HeightMin = 1e10
    LengthMax = np.nan

    for _ in range(N_angle):
        c = np.cos(psi)
        s = np.sin(psi)

        Xwork = (x - Xm) * c + (y - Ym) * s
        Ywork = -(x - Xm) * s + (y - Ym) * c

        Xmin, Xmax = np.min(Xwork), np.max(Xwork)
        Ymin, Ymax = np.min(Ywork), np.max(Ywork)

        LengthCurrent = abs(Xmin) + abs(Xmax)
        HeightCurrent = abs(Ymin) + abs(Ymax)

        if HeightCurrent < HeightMin:
            HeightMin = HeightCurrent
            LengthMax = LengthCurrent

        psi += delta_psi

    if HeightMin <= 0 or not np.isfinite(HeightMin) or not np.isfinite(LengthMax):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    elongation = float(LengthMax / HeightMin)
    bulkiness = float((LengthMax * HeightMin) / area)
    surface = float((perimeter ** 2) / (4.0 * Pi * area))
    circularity = float(2.0 * np.sqrt(Pi * area) / perimeter)

    distances = np.sqrt((x - Xm) ** 2 + (y - Ym) ** 2)
    radius = float(np.max(distances))
    sphericity = 0.0 if radius <= 0 else float((np.sqrt(area / Pi) / radius))

    return elongation, bulkiness, surface, circularity, sphericity


def compute_shape_indices(
    df_xy: pd.DataFrame,
    n_harmonics: int,
    no_sum: int,
    large_limit: float,
    small_limit: float,
    flag_location: int,
    flag_scale: int,
    flag_rotation: int,
    flag_start: int,
    progress_cb=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes shape indices per particle (ID group).

    Returns:
      - results dataframe
      - errors dataframe (rows that failed)
    """
    required = {"ID", "X", "Y"}
    if not required.issubset(df_xy.columns):
        raise ValueError("CSV must have columns: ID, X, Y")

    # Clean & normalize types
    df = df_xy.copy()
    df["ID"] = df["ID"].astype(str)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["ID", "X", "Y"])

    results: List[Dict] = []
    errors: List[Dict] = []

    groups = df.groupby("ID", sort=False)
    ids = list(groups.groups.keys())
    total = len(ids)

    for idx, pid in enumerate(ids, start=1):
        g = groups.get_group(pid)
        x = g["X"].to_numpy(dtype=float)
        y = g["Y"].to_numpy(dtype=float)

        if len(x) < 3:
            errors.append({"ID": pid, "error": "Not enough points (<3)."})
            if progress_cb:
                progress_cb(idx, total, pid)
            continue

        try:
            xmid, ymid = OutlineCentroid(x, y)

            Ax, Bx, Ay, By, scale1, rotate_angle, start_angle = CoefNormalization(
                x, y, xmid, ymid,
                flag_location=flag_location,
                flag_scale=flag_scale,
                flag_rotation=flag_rotation,
                flag_start=flag_start,
                n_harmonics=n_harmonics,
            )

            ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk, Ass1, Ass2, P = ShapeIndicesEF_optimized(
                Ax, Bx, Ay, By,
                no_sum=no_sum,
                large_limit=large_limit,
                small_limit=small_limit,
                xmid=xmid,
                ymid=ymid,
                flag_scale=flag_scale,
                scale1=scale1,
                psi=rotate_angle,
                start_angle=start_angle,
                x=x,
                y=y,
                n_harmonics=n_harmonics,
                flag_rotation=flag_rotation,
            )

            AIg = ComputeAngularity_optimized(Ax, Bx, Ay, By, n_harmonics=no_sum, w=360)
            area = OutlineArea_optimized(x, y)
            perimeter = OutlineCircumference_optimized(x, y)
            elongation, bulkiness, surface, circularity, sphericity = ShapeFactors(x, y, area, perimeter)

            results.append({
                "ID": pid,
                "Kael": k,
                "angularity": AIg,
                "surface_roughness": Uc,
                "asymmetricity1": Ass1,
                "asymmetricity2": Ass2,
                "polygonality": P,
                "area": area,
                "perimeter": perimeter,
                "elongation": elongation,
                "bulkiness": bulkiness,
                "surface": surface,
                "circularity": circularity,
                "sphericity": sphericity,
            })

        except Exception as e:
            errors.append({"ID": pid, "error": repr(e)})

        if progress_cb:
            progress_cb(idx, total, pid)

    return pd.DataFrame(results), pd.DataFrame(errors)


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="Shape Analysis (Optimized)", layout="wide")
st.title("🔬 Shape Analysis (Optimized) — upload ID/X/Y → get shape indices")

if not SCIPY_SPECIAL_AVAILABLE:
    st.error("This app requires SciPy (scipy.special.elliprf/elliprd). Please install SciPy.")
    st.stop()

# Sidebar inputs
st.sidebar.header("1) Upload coordinate CSV")

uploaded_csv = st.sidebar.file_uploader(
    "CSV with columns: ID, X, Y (long format; multiple rows per ID)",
    type=["csv"],
    accept_multiple_files=False,
)

st.sidebar.header("2) Parameters (advanced)")
with st.sidebar.expander("Fourier / indices settings", expanded=False):
    n_harmonics = st.number_input("No. harmonics (EFD)", min_value=1, max_value=128, value=16, step=1)
    no_sum = st.number_input("No_sum (used in indices)", min_value=1, max_value=128, value=16, step=1)
    large_limit = st.number_input("LargeLimit", min_value=1.0, value=100.0, step=1.0)
    small_limit = st.number_input("SmallLimit", min_value=0.0, value=1e-5, format="%.8f")

with st.sidebar.expander("Normalization flags", expanded=False):
    flag_location = st.selectbox("FlagLocation (center to centroid)", [0, 1], index=1)
    flag_scale = st.selectbox("FlagScale (size invariant)", [0, 1], index=1)
    flag_rotation = st.selectbox("FlagRotation (rotation invariant)", [0, 1], index=1)
    flag_start = st.selectbox("FlagStart (start point invariant)", [0, 1], index=1)

st.sidebar.header("3) Optional images (for comparison tab)")
uploaded_images = st.sidebar.file_uploader(
    "Upload particle images (filenames should match ID, e.g. 123.png)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# Load data
if uploaded_csv is None:
    st.info("⬅️ Upload a CSV to start.")
    st.stop()

@st.cache_data
def load_xy(file_bytes: bytes) -> pd.DataFrame:
    """Robust CSV loader: tries comma/semicolon/tab and normalizes column names."""
    raw = io.BytesIO(file_bytes)
    # Try common separators first; fall back to python engine sniffing.
    tries = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": "	", "encoding": "utf-8-sig"},
    ]
    last_err = None
    for kw in tries:
        try:
            raw.seek(0)
            df = pd.read_csv(raw, **kw)
            if df.shape[1] >= 3:
                break
        except Exception as e:
            last_err = e
            continue
    else:
        try:
            raw.seek(0)
            df = pd.read_csv(raw, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            raise ValueError(f"Could not read CSV. Last error: {last_err!r}, fallback error: {e!r}")

    # Normalize column names (case/whitespace tolerant)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    if "id" in lower: rename[lower["id"]] = "ID"
    if "x" in lower: rename[lower["x"]] = "X"
    if "y" in lower: rename[lower["y"]] = "Y"
    df = df.rename(columns=rename)
    return df

df_xy = load_xy(uploaded_csv.getvalue())

# Build image dict (optional)
def build_image_dict(files) -> Dict[str, Image.Image]:
    img_dict: Dict[str, Image.Image] = {}
    for f in files or []:
        stem, _ = os.path.splitext(f.name)
        try:
            img_dict[str(stem)] = Image.open(f).copy()
        except Exception:
            continue
    return img_dict

image_dict = build_image_dict(uploaded_images)

# Tabs
tab_run, tab_results, tab_compare, tab_analytics = st.tabs(
    ["▶️ Run analysis", "🧾 Results table", "🆚 Compare particles", "📊 Analytics"]
)

with tab_run:
    st.subheader("Input preview")
    st.dataframe(df_xy.head(50), use_container_width=True)

    if not {"ID", "X", "Y"}.issubset(df_xy.columns):
        st.error("Your CSV must contain columns: ID, X, Y.")
        st.stop()

    n_particles = df_xy["ID"].nunique(dropna=True)
    st.write(f"Detected **{n_particles}** particles (unique IDs).")

    # Points-per-particle quality check
    counts = df_xy.groupby("ID").size().sort_values(ascending=False)
    small = counts[counts < 10]
    with st.expander("Points per particle (quality check)", expanded=False):
        st.dataframe(counts.reset_index(name="n_points").head(50), use_container_width=True)
        st.caption("Tip: Very small outlines (e.g., <10 points) may cause unstable indices. You can filter them before running.")
        if len(small):
            st.warning(f"{len(small)} particle(s) have <10 points. They may fail or produce noisy results.")

    colA, colB = st.columns([1, 2])
    with colA:
        run = st.button("Run optimized shape analysis", type="primary")
    with colB:
        st.caption("Tip: run once, then use the other tabs to explore & download results.")

    if run:
        st.session_state.pop("results_df", None)
        st.session_state.pop("errors_df", None)
        progress = st.progress(0)
        status = st.empty()

        t0 = time.time()

        def progress_cb(i, total, pid):
            progress.progress(int(100 * i / max(total, 1)))
            status.write(f"Processing **{pid}** ({i}/{total})")

        with st.spinner("Computing shape indices..."):
            results_df, errors_df = compute_shape_indices(
                df_xy=df_xy,
                n_harmonics=int(n_harmonics),
                no_sum=int(no_sum),
                large_limit=float(large_limit),
                small_limit=float(small_limit),
                flag_location=int(flag_location),
                flag_scale=int(flag_scale),
                flag_rotation=int(flag_rotation),
                flag_start=int(flag_start),
                progress_cb=progress_cb,
            )

        dt = time.time() - t0
        progress.empty()
        status.empty()

        st.session_state["results_df"] = results_df
        st.session_state["errors_df"] = errors_df

        st.success(f"Done. Computed {len(results_df)} particles in {dt:.2f}s.")
        if len(errors_df):
            st.warning(f"{len(errors_df)} particles failed (see Results tab).")

with tab_results:
    results_df = st.session_state.get("results_df")
    errors_df = st.session_state.get("errors_df")

    if results_df is None:
        st.info("Run the analysis first (▶️ Run analysis tab).")
    else:
        st.subheader("Results")
        st.dataframe(results_df, use_container_width=True)

        # Downloads
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results CSV",
            data=csv_bytes,
            file_name="shape_indices_results_optimized.csv",
            mime="text/csv",
        )

        if errors_df is not None and len(errors_df):
            st.subheader("Errors")
            st.dataframe(errors_df, use_container_width=True)
            err_bytes = errors_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download error report CSV",
                data=err_bytes,
                file_name="shape_indices_errors.csv",
                mime="text/csv",
            )

with tab_compare:
    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty:
        st.info("Run the analysis first (▶️ Run analysis tab).")
    else:
        st.subheader("Compare two particles (uses computed results)")
        ids = results_df["ID"].astype(str).tolist()

        c1, c2 = st.columns(2)
        with c1:
            pid1 = st.selectbox("Particle 1", ids, index=0, key="pid1")
        with c2:
            pid2 = st.selectbox("Particle 2", ids, index=min(1, len(ids) - 1), key="pid2")

        numeric_cols = results_df.select_dtypes(include="number").columns.tolist()

        def display_particle(pid: str):
            st.markdown(f"### Particle {pid}")
            img = image_dict.get(str(pid))
            if img is not None:
                st.image(img, caption=f"Image: {pid}", use_container_width=True)
            else:
                st.info("No image uploaded for this ID (optional).")
            row = results_df[results_df["ID"].astype(str) == str(pid)]
            st.dataframe(row, use_container_width=True)

        left, right = st.columns(2)
        with left:
            display_particle(pid1)
        with right:
            display_particle(pid2)

        if numeric_cols:
            compare_cols = st.multiselect(
                "Select properties to compare",
                options=numeric_cols,
                default=numeric_cols[: min(6, len(numeric_cols))],
            )
            if compare_cols:
                comp = results_df[results_df["ID"].astype(str).isin([str(pid1), str(pid2)])][["ID"] + compare_cols]
                melted = comp.melt(id_vars="ID", var_name="Property", value_name="Value")
                fig = px.bar(melted, x="Property", y="Value", color="ID", barmode="group",
                             title="Particle property comparison")
                st.plotly_chart(fig, use_container_width=True)

with tab_analytics:
    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty:
        st.info("Run the analysis first (▶️ Run analysis tab).")
    else:
        st.subheader("Analytics & statistics")
        numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                st.markdown("#### Summary statistics")
                desc = results_df[numeric_cols].describe().T
                st.dataframe(desc, use_container_width=True)

            with col_stat2:
                st.markdown("#### Extra statistics")
                extra = pd.DataFrame({
                    "skew": results_df[numeric_cols].skew(),
                    "kurtosis": results_df[numeric_cols].kurt(),
                    "missing_count": results_df[numeric_cols].isna().sum(),
                    "missing_pct": results_df[numeric_cols].isna().mean() * 100.0,
                })
                st.dataframe(extra, use_container_width=True)

            st.markdown("---")

            st.markdown("#### Scatter plot")
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols, index=0, key="sc_x")
                y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="sc_y")
                fig_sc = px.scatter(results_df, x=x_col, y=y_col, hover_name="ID", title="Scatter plot")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns.")

            st.markdown("---")

            st.markdown("#### Distribution + KDE (if SciPy available)")
            dist_col = st.selectbox("Column", numeric_cols, index=0, key="dist_col")
            data = results_df[dist_col].dropna().values

            if data.size > 0:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=data, histnorm="probability density", nbinsx=30, name="Histogram", opacity=0.6
                ))
                if SCIPY_AVAILABLE and data.size > 1:
                    xs = np.linspace(float(np.min(data)), float(np.max(data)), 400)
                    kde = gaussian_kde(data)
                    fig_dist.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name="KDE"))
                fig_dist.update_layout(
                    title=f"Distribution: {dist_col}", xaxis_title=dist_col, yaxis_title="Density", barmode="overlay"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No data in this column.")

            st.markdown("---")

            st.markdown("#### Correlation matrix (Pearson)")
            corr = results_df[numeric_cols].corr(method="pearson")
            st.dataframe(corr, use_container_width=True)
            fig_corr = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                 title="Correlation matrix (Pearson)")
            fig_corr.update_xaxes(side="bottom")
            st.plotly_chart(fig_corr, use_container_width=True)
