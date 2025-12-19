# Shape Analysis Streamlit App (single file, v4)
# -------------------------------------------------
# Features:
# 1) Single particle: outline reconstruction vs harmonics + ellipse (N=1) + equal-area circle
# 2) Sensitivity: Assymetry & Polygonality vs harmonics (2..20 by default), clear boxplots
# 3) Whole sample: batch analysis + downloads + analytics (filters, outliers, scatter, histogram, correlation)
#
# Install:
#   pip install streamlit numpy pandas scipy plotly pillow matplotlib
# Optional (Page 1):
#   pip install spatial_efd
#
# Run:
#   python -m streamlit run ShapeAnalysis_Streamlit_App_v4_singlefile.py

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

# Optional (Page 1)
try:
    import spatial_efd.spatial_efd as spatial_efd
    SPATIAL_EFD_AVAILABLE = True
except Exception:
    SPATIAL_EFD_AVAILABLE = False

# Required (core math)
try:
    from scipy.special import elliprd, elliprf
    SCIPY_SPECIAL_AVAILABLE = True
except Exception:
    SCIPY_SPECIAL_AVAILABLE = False


# =============================================================================
# Embedded instruction images (A–F) — labeled & upscaled
# =============================================================================


# =============================================================================
# Instruction figures
# =============================================================================
# To keep this file readable, instruction images are NOT embedded as base64.
# If you want figures in the app, place PNG files next to this script in:
#   ./assets/instruction_A.png ... instruction_F.png
#
# (This keeps the code clean and still supports high-quality figures.)
ASSETS_DIR = Path(__file__).parent / "assets"
INSTRUCTION_LABELS = ["A", "B", "C", "D", "E", "F"]

def render_instruction_figures():
    """Compact 3×2 grid inside an expander (loads from ./assets if present)."""
    existing = []
    for lab in INSTRUCTION_LABELS:
        p = ASSETS_DIR / f"instruction_{lab}.png"
        if p.exists():
            existing.append((lab, p))
    if not existing:
        st.caption("Instruction figures: add files to ./assets (instruction_A.png ... instruction_F.png) to display them.")
        return
    with st.expander("📌 Instruction figures (click to expand)", expanded=False):
        cols = st.columns(3)
        for i, (lab, p) in enumerate(existing):
            with cols[i % 3]:
                st.image(str(p), use_container_width=True)


# =============================================================================
# Robust CSV loader + ID mapping
# =============================================================================
def load_xy_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Robust loader: supports comma/semicolon/tab separators and column-name variants.
    Output columns: ID, X, Y
    """
    df = None
    for sep in [",", ";", "\t", None]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
            if df.shape[1] >= 3:
                break
        except Exception:
            df = None

    if df is None or df.empty:
        raise ValueError("Could not read CSV. Check delimiter/encoding. Expected columns: ID, X, Y.")

    # normalize column names
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    lower_map = {c.lower().strip(): c for c in df.columns}

    def pick(opts: List[str]) -> str | None:
        for o in opts:
            if o in lower_map:
                return lower_map[o]
        return None

    id_col = pick(["id", "particle_id", "particle", "pid"])
    x_col = pick(["x", "xcoord", "x_coordinate"])
    y_col = pick(["y", "ycoord", "y_coordinate"])

    # fallback: first 3 columns
    if id_col is None or x_col is None or y_col is None:
        id_col, x_col, y_col = df.columns[:3]

    out = df[[id_col, x_col, y_col]].copy()
    out.columns = ["ID", "X", "Y"]
    out["ID"] = out["ID"].astype(str).str.strip()
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["ID", "X", "Y"])
    return out


def make_id_mapping(original_ids: List[str], mode: str) -> Dict[str, str]:
    """
    mode:
      - "Sequential P001"
      - "Original"
    """
    if mode == "Original":
        return {oid: oid for oid in original_ids}
    mapping = {}
    for i, oid in enumerate(original_ids, start=1):
        mapping[oid] = f"P{i:03d}"
    return mapping


# =============================================================================
# Optimized shape analysis engine (same logic as your optimized pipeline)
# =============================================================================
def ELLE(ak: float) -> float:
    if not SCIPY_SPECIAL_AVAILABLE:
        raise RuntimeError("SciPy (scipy.special.elliprf/elliprd) is required.")
    ak = float(np.clip(ak, 0.0, 1.0))
    s = 1.0
    cc = 0.0
    Q = (1.0 - s * ak) * (1.0 + s * ak)
    return s * (elliprf(cc, Q, 1.0) - ((s * ak) * (s * ak)) * elliprd(cc, Q, 1.0) / 3.0)


def OutlineCentroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.mean(y))


def OutlineArea(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def OutlinePerimeter(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def ComputeEllFourierCoef(x: np.ndarray, y: np.ndarray, n_harmonics: int):
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

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    dx = x_next - x
    dy = y_next - y
    dt = np.sqrt(dx * dx + dy * dy)

    rx = np.where(dt != 0.0, dx / dt, 0.0)
    ry = np.where(dt != 0.0, dy / dt, 0.0)

    t_curr = np.cumsum(dt)
    t_prev = np.concatenate(([0.0], t_curr[:-1]))
    T = float(t_curr[-1]) if len(t_curr) else 0.0

    # DC terms
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
        Ax[0] = float(x[0] + Ax0_int / T)
        Ay[0] = float(y[0] + Ay0_int / T)
    else:
        Ax[0] = float(x[0])
        Ay[0] = float(y[0])

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


def FourierCoefNormalization(Ax, Bx, Ay, By, flag_scale, flag_rotation, flag_start):
    Ax = Ax.copy(); Bx = Bx.copy(); Ay = Ay.copy(); By = By.copy()
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


def CoefNormalization(x, y, xmid, ymid, flag_location, flag_scale, flag_rotation, flag_start, n_harmonics):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if flag_location == 1:
        x0 = x - xmid
        y0 = y - ymid
    else:
        x0 = x
        y0 = y
    Ax, Bx, Ay, By = ComputeEllFourierCoef(x0, y0, n_harmonics)
    return FourierCoefNormalization(Ax, Bx, Ay, By, flag_scale, flag_rotation, flag_start)


def ShapeIndex(Ax, Bx, Ay, By, flag_rotation, psi, xmid, ymid, scale1, x, y, n0_sum, large_number):
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

    # Sharpness (vectorized)
    xp1 = np.roll(xp, -1); yp1 = np.roll(yp, -1)
    dx = xp1 - xp; dy = yp1 - yp
    di = np.full_like(dx, 1000.0, dtype=np.float64)
    mask = np.abs(dx) > 1e-30
    np.divide(dy, dx, out=di, where=mask)

    jj = np.arange(36, dtype=np.float64)
    theta_jj = (2.0 * np.pi / 36.0) * jj
    cos_jj = np.cos(theta_jj); sin_jj = np.sin(theta_jj)

    di2 = di[:, None]
    numerator = di2 * cos_jj - sin_jj
    denominator = cos_jj + di2 * sin_jj
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    aver_sum = np.sum(ratio, axis=1)
    aver = np.arctan(aver_sum / 36.0)

    # S1
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

    # S2
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


def ShapeIndicesEF(Ax, Bx, Ay, By, no_sum, large_limit, small_limit,
                   xmid, ymid, flag_scale, scale1, psi, start_angle,
                   x, y, n_harmonics, flag_rotation):
    ae, be, k, Le, rc, Uc, S1, S2, Bk1, Bk2, Ask, Bsk = ShapeIndex(
        Ax, Bx, Ay, By,
        flag_rotation=flag_rotation,
        psi=psi,
        xmid=xmid, ymid=ymid,
        scale1=1.0 if flag_scale == 0 else scale1,
        x=x, y=y, n0_sum=no_sum,
        large_number=large_limit
    )

    Ax2, Bx2, Ay2, By2 = ComputeEllFourierCoef(x, y, n_harmonics)

    no_ass = min(n_harmonics, no_sum)
    Ax_s = Ax2[1:no_ass + 1]; Bx_s = Bx2[1:no_ass + 1]
    Ay_s = Ay2[1:no_ass + 1]; By_s = By2[1:no_ass + 1]

    abs_Ax = np.abs(Ax_s); abs_Bx = np.abs(Bx_s)
    abs_Ay = np.abs(Ay_s); abs_By = np.abs(By_s)

    cond_bx = (abs_Bx > small_limit) & (abs_Ax > small_limit)
    cond_ay = (abs_Ay > small_limit) & (abs_By > small_limit)

    ratio_bx = np.divide(abs_Bx, abs_Ax, out=np.zeros_like(abs_Bx), where=cond_bx)
    ratio_ay = np.divide(abs_Ay, abs_By, out=np.zeros_like(abs_Ay), where=cond_ay)

    SumBx = float(np.sum(ratio_bx))
    SumAy = float(np.sum(ratio_ay))
    Ass2 = float(np.sqrt(SumBx * SumAy)) if (SumBx >= 0 and SumAy >= 0) else np.nan

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

    # Polygonality
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


def ComputeAngularity(Ax, Bx, Ay, By, n_harmonics: int, w: int = 360) -> float:
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


def ShapeFactors(x, y, area=None, perimeter=None):
    Pi = np.pi
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if area is None:
        area = OutlineArea(x, y)
    if perimeter is None:
        perimeter = OutlinePerimeter(x, y)

    if area <= 0 or perimeter <= 0 or not np.isfinite(area) or not np.isfinite(perimeter):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    Xm, Ym = float(np.mean(x)), float(np.mean(y))

    N_angle = 180
    delta_psi = Pi / (N_angle - 1)
    psi = 0.0
    HeightMin = 1e10
    LengthMax = np.nan

    for _ in range(N_angle):
        c = np.cos(psi); s = np.sin(psi)
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


@dataclass(frozen=True)
class ShapeParams:
    n_harmonics: int = 16
    no_sum: int = 16
    large_limit: float = 100.0
    small_limit: float = 1e-5
    flag_location: int = 1
    flag_scale: int = 1
    flag_rotation: int = 1
    flag_start: int = 1


def compute_shape_indices(df_xy: pd.DataFrame, params: ShapeParams, progress_cb=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not SCIPY_SPECIAL_AVAILABLE:
        raise RuntimeError("This computation requires SciPy (scipy.special.elliprf/elliprd).")

    df = df_xy.copy()
    df["ID"] = df["ID"].astype(str)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["ID", "X", "Y"])

    results = []
    errors = []

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
                flag_location=params.flag_location,
                flag_scale=params.flag_scale,
                flag_rotation=params.flag_rotation,
                flag_start=params.flag_start,
                n_harmonics=params.n_harmonics,
            )

            ae, be, k, Le, rc, Uc, S1, S2, *_rest, Ass1, Ass2, P = ShapeIndicesEF(
                Ax, Bx, Ay, By,
                no_sum=params.no_sum,
                large_limit=params.large_limit,
                small_limit=params.small_limit,
                xmid=xmid, ymid=ymid,
                flag_scale=params.flag_scale,
                scale1=scale1,
                psi=rotate_angle,
                start_angle=start_angle,
                x=x, y=y,
                n_harmonics=params.n_harmonics,
                flag_rotation=params.flag_rotation,
            )

            angularity = ComputeAngularity(Ax, Bx, Ay, By, n_harmonics=params.no_sum, w=360)
            area = OutlineArea(x, y)
            perimeter = OutlinePerimeter(x, y)
            elongation, bulkiness, surface, circularity, sphericity = ShapeFactors(x, y, area, perimeter)

            results.append({
                "Original_ID": pid,
                "Elongation_ratio": k,
                "Angularity": angularity,
                "Surface_roughness": Uc,
                "Assymetry": Ass1,
                "Assymetry_normalized": Ass2,
                "Polygonality": P,
                "Area": area,
                "Perimeter": perimeter,
                "Elongation": elongation,
                "Bulkiness": bulkiness,
                "Surface": surface,
                "Circularity": circularity,
                "Sphericity": sphericity,
            })
        except Exception as e:
            errors.append({"ID": pid, "error": repr(e)})

        if progress_cb:
            progress_cb(idx, total, pid)

    return pd.DataFrame(results), pd.DataFrame(errors)


# =============================================================================
# Page helpers
# =============================================================================
def points_per_particle(df_xy: pd.DataFrame) -> pd.DataFrame:
    return df_xy.groupby("ID", sort=False).size().reset_index(name="n_points").sort_values("n_points")


def plot_reconstruction_spatial_efd(x: np.ndarray, y: np.ndarray, N: int):
    """Original vs reconstruction using spatial_efd, compact figure."""
    if len(x) >= 3 and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0]); y = np.append(y, y[0])

    coeffs = spatial_efd.CalculateEFD(x, y, N)
    locus = spatial_efd.calculate_dc_coefficients(x, y)
    xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=N, locus=locus)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=170)
    ax.plot(x, y, linewidth=1.6, label="Original")
    ax.plot(xt, yt, linewidth=1.6, label=f"Reconstruction (N={N})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Outline reconstruction", fontsize=11)
    return fig


def plot_ellipse_and_circle(x: np.ndarray, y: np.ndarray):
    """Approximate ellipse via N=1 + equal-area circle."""
    if len(x) >= 3 and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0]); y = np.append(y, y[0])

    coeffs1 = spatial_efd.CalculateEFD(x, y, 1)
    locus = spatial_efd.calculate_dc_coefficients(x, y)
    xt1, yt1 = spatial_efd.inverse_transform(coeffs1, harmonic=1, locus=locus)

    area = OutlineArea(x, y)
    cx, cy = OutlineCentroid(x, y)
    r = np.sqrt(area / np.pi) if area > 0 else 0.0

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=170)
    ax.plot(x, y, linewidth=1.5, label="Original")
    ax.plot(xt1, yt1, linewidth=1.8, label="Approx. ellipse (N=1)")
    if r > 0:
        ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=1.8, linestyle="--", label="Equal-area circle"))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Ellipse & equal-area circle", fontsize=11)
    return fig


# =============================================================================
# Streamlit app
# =============================================================================
st.set_page_config(page_title="Shape Analysis", layout="wide")
st.title("🔬 Shape Analysis")

uploaded = st.sidebar.file_uploader("Upload CSV (ID, X, Y)", type=["csv"])

st.sidebar.markdown("---")
id_mode = st.sidebar.selectbox("Particle ID display", ["Sequential P001", "Original"], index=0)

st.sidebar.markdown("---")
page = st.sidebar.radio("Pages", ["1) Single particle", "2) Sensitivity analysis", "3) Whole sample"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Computation parameters")

with st.sidebar.expander("Fourier / index settings", expanded=False):
    n_harmonics = st.number_input("No. harmonics (EFD)", min_value=1, max_value=128, value=16, step=1)
    no_sum = st.number_input("No_sum (used in indices)", min_value=1, max_value=128, value=16, step=1)
    large_limit = st.number_input("LargeLimit", min_value=1.0, value=100.0, step=1.0)
    small_limit = st.number_input("SmallLimit", min_value=0.0, value=1e-5, format="%.8f")

with st.sidebar.expander("Normalization flags", expanded=False):
    flag_location = st.selectbox("FlagLocation (center to centroid)", [0, 1], index=1)
    flag_scale = st.selectbox("FlagScale (size invariant)", [0, 1], index=1)
    flag_rotation = st.selectbox("FlagRotation (rotation invariant)", [0, 1], index=1)
    flag_start = st.selectbox("FlagStart (start point invariant)", [0, 1], index=1)

params = ShapeParams(
    n_harmonics=int(n_harmonics),
    no_sum=int(no_sum),
    large_limit=float(large_limit),
    small_limit=float(small_limit),
    flag_location=int(flag_location),
    flag_scale=int(flag_scale),
    flag_rotation=int(flag_rotation),
    flag_start=int(flag_start),
)

if uploaded is None:
    st.info("⬅️ Upload a CSV to start.")
    render_instruction_figures()
    st.stop()

if not SCIPY_SPECIAL_AVAILABLE:
    st.error("SciPy is required for this app: `pip install scipy`.")
    st.stop()

@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes) -> pd.DataFrame:
    return load_xy_csv(file_bytes)

df_xy = _load(uploaded.getvalue())
if df_xy.empty:
    st.error("CSV loaded but has no valid rows for ID/X/Y.")
    st.stop()

original_ids = df_xy["ID"].dropna().astype(str).unique().tolist()
id_map = make_id_mapping(original_ids, id_mode)
reverse_map = {v: k for k, v in id_map.items()}

st.caption(f"Detected **{df_xy['ID'].nunique()}** particles and **{len(df_xy)}** outline points.")

# ---------------- Page 1 ----------------
if page.startswith("1"):
    st.header("1) Single particle")
    render_instruction_figures()

    pts = points_per_particle(df_xy)
    if int(pts["n_points"].min()) < 20:
        st.warning("Some particles have very few points (<20). Reconstruction may be unstable.")

    display_ids = [id_map[o] for o in original_ids]
    chosen_display = st.selectbox("Choose particle", display_ids, index=0)
    chosen_original = reverse_map.get(chosen_display, chosen_display)

    g = df_xy[df_xy["ID"].astype(str) == str(chosen_original)]
    x = g["X"].to_numpy(dtype=float)
    y = g["Y"].to_numpy(dtype=float)

    if len(x) < 3:
        st.error("This particle has <3 points.")
        st.stop()

    if not SPATIAL_EFD_AVAILABLE:
        st.error("Page 1 needs `spatial_efd`. Install: `pip install spatial_efd`.")
        st.stop()

    nyq = int(spatial_efd.Nyquist(x))
    c1, c2 = st.columns([1, 2])
    with c1:
        N = st.slider("Number of harmonics (N)", min_value=1, max_value=max(1, nyq), value=min(10, max(1, nyq)))
        st.caption(f"Nyquist limit: **{nyq}**")
    with c2:
        st.caption("Small N → smooth outline; larger N → more detail/noise.")

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_reconstruction_spatial_efd(x, y, int(N)), use_container_width=True)
    with right:
        st.pyplot(plot_ellipse_and_circle(x, y), use_container_width=True)

# ---------------- Page 2 ----------------
elif page.startswith("2"):
    st.header("2) Sensitivity analysis")
    st.caption("Assymetry and Polygonality distributions across particles for selected numbers of harmonics.")

    # Pick harmonics explicitly (default like your example)
    max_h = 50
    default_N = [1, 2, 5, 10, 15, 20]
    options = list(range(1, max_h + 1))
    chosen_N = st.multiselect(
        "Select harmonics to evaluate",
        options=options,
        default=[n for n in default_N if n <= max_h],
    )

    chosen_N = sorted(set(int(n) for n in chosen_N))
    if not chosen_N:
        st.info("Select at least one harmonic number.")
        st.stop()

    st.caption(f"Selected: {', '.join(map(str, chosen_N))}")

    @st.cache_data(show_spinner=False)
    def _compute_for_N(file_hash: str, df_xy: pd.DataFrame, params_base: ShapeParams, N: int) -> pd.DataFrame:
        p = ShapeParams(
            n_harmonics=int(N),
            no_sum=int(N),
            large_limit=params_base.large_limit,
            small_limit=params_base.small_limit,
            flag_location=params_base.flag_location,
            flag_scale=params_base.flag_scale,
            flag_rotation=params_base.flag_rotation,
            flag_start=params_base.flag_start,
        )
        res, _err = compute_shape_indices(df_xy, p)
        if res.empty:
            return pd.DataFrame()
        out = res[["Original_ID", "Assymetry", "Polygonality"]].copy()
        out["Harmonics"] = int(N)
        return out

    run = st.button("Run sensitivity", type="primary")
    if run:
        import hashlib
        file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

        prog = st.progress(0)
        status = st.empty()

        rows = []
        total = len(chosen_N)
        for i, N in enumerate(chosen_N, start=1):
            status.write(f"Computing N = {N}  ({i}/{total})")
            try:
                part = _compute_for_N(file_hash, df_xy, params, int(N))
                if not part.empty:
                    rows.append(part)
            finally:
                prog.progress(int(100 * i / max(total, 1)))

        status.empty()
        prog.empty()

        if not rows:
            st.error("No sensitivity data produced.")
            st.stop()

        sens = pd.concat(rows, ignore_index=True)
        sens["Particle_ID"] = sens["Original_ID"].map(id_map).fillna(sens["Original_ID"])

        st.subheader("Assymetry vs harmonics")
        figA = px.box(sens, x="Harmonics", y="Assymetry", points="outliers", hover_data=["Particle_ID"])
        figA.update_layout(xaxis_title="Number of harmonics", yaxis_title="Assymetry")
        st.plotly_chart(figA, use_container_width=True)

        st.subheader("Polygonality vs harmonics")
        figP = px.box(sens, x="Harmonics", y="Polygonality", points="outliers", hover_data=["Particle_ID"])
        figP.update_layout(xaxis_title="Number of harmonics", yaxis_title="Polygonality")
        st.plotly_chart(figP, use_container_width=True)

        st.markdown("### Sensitivity data")
        st.dataframe(sens.sort_values(["Harmonics", "Particle_ID"]), use_container_width=True)

        st.download_button(
            "⬇️ Download sensitivity CSV",
            data=sens.to_csv(index=False).encode("utf-8"),
            file_name="sensitivity_assymetry_polygonality.csv",
            mime="text/csv",
        )

# ---------------- Page 3 ----------------
else:
    st.header("3) Whole sample")

    with st.expander("Outline quality check", expanded=False):
        st.dataframe(points_per_particle(df_xy), use_container_width=True)
        st.caption("Low point counts can cause unstable indices.")

    run = st.button("Run shape analysis", type="primary")
    if run:
        st.session_state.pop("results_df", None)
        st.session_state.pop("errors_df", None)

        progress = st.progress(0)
        status = st.empty()

        t0 = time.time()

        def progress_cb(i, total, pid):
            progress.progress(int(100 * i / max(total, 1)))
            status.write(f"Processing {id_map.get(pid, pid)} ({i}/{total})")

        with st.spinner("Computing shape indices..."):
            results_df, errors_df = compute_shape_indices(df_xy, params, progress_cb=progress_cb)

        dt = time.time() - t0
        progress.empty()
        status.empty()

        results_df = results_df.copy()
        results_df["Particle_ID"] = results_df["Original_ID"].map(id_map).fillna(results_df["Original_ID"])
        results_df = results_df[["Particle_ID"] + [c for c in results_df.columns if c != "Particle_ID"]]

        st.session_state["results_df"] = results_df
        st.session_state["errors_df"] = errors_df

        st.success(f"Done. Computed {len(results_df)} particles in {dt:.2f}s.")
        if errors_df is not None and len(errors_df):
            st.warning(f"{len(errors_df)} particles failed (see Errors).")

    results_df = st.session_state.get("results_df")
    errors_df = st.session_state.get("errors_df")

    if results_df is None:
        st.info("Click **Run shape analysis** to compute results.")
        st.stop()

    st.subheader("Results")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "⬇️ Download results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="shape_indices_results.csv",
        mime="text/csv",
    )

    if errors_df is not None and len(errors_df):
        with st.expander("Errors", expanded=False):
            st.dataframe(errors_df, use_container_width=True)
            st.download_button(
                "⬇️ Download errors CSV",
                data=errors_df.to_csv(index=False).encode("utf-8"),
                file_name="shape_indices_errors.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.subheader("Analytics tools")

    numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found.")
        st.stop()

    # Filters
    with st.expander("Filters", expanded=False):
        id_search = st.text_input("Search Particle_ID", value="")
        filtered = results_df.copy()
        if id_search.strip():
            filtered = filtered[filtered["Particle_ID"].astype(str).str.contains(id_search.strip(), case=False, na=False)]

        filter_cols = st.multiselect("Numeric columns to filter", numeric_cols, default=[])
        for c in filter_cols:
            col_vals = filtered[c].dropna()
            if col_vals.empty:
                continue
            lo, hi = float(col_vals.min()), float(col_vals.max())
            if lo == hi:
                continue
            rng = st.slider(f"{c} range", min_value=lo, max_value=hi, value=(lo, hi))
            filtered = filtered[(filtered[c] >= rng[0]) & (filtered[c] <= rng[1])]

        st.caption(f"Filtered rows: {len(filtered)} / {len(results_df)}")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "⬇️ Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="shape_indices_filtered.csv",
            mime="text/csv",
        )

    # Outliers (IQR)
    with st.expander("Outliers (IQR)", expanded=False):
        out_col = st.selectbox("Column for outlier detection", numeric_cols, index=0)
        k_iqr = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        vals = results_df[out_col].dropna()
        if len(vals) >= 4:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - k_iqr * iqr, q3 + k_iqr * iqr
            outliers = results_df[(results_df[out_col] < lo) | (results_df[out_col] > hi)]
            st.caption(f"Outliers: {len(outliers)} (thresholds: {lo:.4g} .. {hi:.4g})")
            st.dataframe(outliers[["Particle_ID", "Original_ID", out_col]], use_container_width=True)
        else:
            st.info("Not enough data for IQR outliers.")

    # Charts
    st.markdown("### Charts")
    c1, c2 = st.columns(2)
    with c1:
        xcol = st.selectbox("X axis", numeric_cols, index=0)
    with c2:
        ycol = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))

    st.plotly_chart(
        px.scatter(filtered, x=xcol, y=ycol, hover_name="Particle_ID", title="Scatter plot (filtered)"),
        use_container_width=True,
    )

    dist_col = st.selectbox("Distribution column", numeric_cols, index=0)
    st.plotly_chart(
        px.histogram(filtered, x=dist_col, nbins=30, title=f"Distribution: {dist_col}"),
        use_container_width=True,
    )

    with st.expander("Correlation matrix", expanded=False):
        corr = filtered[numeric_cols].corr(method="pearson")
        fig_corr = px.imshow(corr, aspect="auto", zmin=-1, zmax=1, title="Correlation (Pearson)")
        fig_corr.update_xaxes(side="bottom")
        st.plotly_chart(fig_corr, use_container_width=True)
