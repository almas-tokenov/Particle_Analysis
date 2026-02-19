from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

# Optional: spatial_efd (Module 1)
try:
    import spatial_efd  # type: ignore
    SPATIAL_EFD_AVAILABLE = True
except Exception:
    SPATIAL_EFD_AVAILABLE = False

# Optional: SciPy special (ellipse perimeter). Fallback to Ramanujan if unavailable.
try:
    from scipy.special import elliprd, elliprf  # type: ignore
    SCIPY_SPECIAL_AVAILABLE = True
except Exception:
    SCIPY_SPECIAL_AVAILABLE = False


# =============================================================================
# Paths + instruction images
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIRS = [
    APP_DIR / "assets",
    APP_DIR / "Instruction images",
    APP_DIR,
]


def _resolve_image(path_or_name: str) -> Optional[Path]:
    """Resolve an image by absolute path or by filename in ./assets or ./Instruction images."""
    try:
        p = Path(path_or_name)
        if p.exists():
            return p
    except Exception:
        pass

    name = Path(str(path_or_name)).name
    for base in ASSETS_DIRS:
        cand = base / name
        if cand.exists():
            return cand
    return None


def show_images(paths: List[str], title: str, expanded: bool, ncols: int = 2) -> None:
    """Image grid inside an expander. Ignores missing images."""
    resolved: List[Path] = []
    for s in paths:
        p = _resolve_image(s)
        if p is not None:
            resolved.append(p)
    if not resolved:
        return

    with st.expander(title, expanded=expanded):
        cols = st.columns(ncols)
        for i, p in enumerate(resolved):
            with cols[i % ncols]:
                st.image(str(p), use_container_width=True)


# =============================================================================
# CSV loader
# =============================================================================
def load_xy_csv(file_bytes: bytes) -> pd.DataFrame:
    """Robust loader for ID, X, Y (handles , ; tab)."""
    df = None
    for sep in [",", ";", "\t", None]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
            if df is not None and df.shape[1] >= 3:
                break
        except Exception:
            df = None

    if df is None or df.empty:
        raise ValueError("Could not read CSV. Expected columns: ID, X, Y.")

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    lower_map = {c.lower().strip(): c for c in df.columns}

    def pick(opts: List[str]) -> Optional[str]:
        for o in opts:
            if o in lower_map:
                return lower_map[o]
        return None

    id_col = pick(["id", "particle", "particle_id", "pid", "label"])
    x_col = pick(["x", "xcoord", "x_coord", "x-coordinate"])
    y_col = pick(["y", "ycoord", "y_coord", "y-coordinate"])

    if id_col is None or x_col is None or y_col is None:
        cols = list(df.columns[:3])
        id_col, x_col, y_col = cols[0], cols[1], cols[2]

    out = df[[id_col, x_col, y_col]].copy()
    out.columns = ["ID", "X", "Y"]
    out["ID"] = out["ID"].astype(str).str.strip()
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["ID", "X", "Y"]).reset_index(drop=True)
    return out


def make_id_mapping(original_ids: List[str]) -> Dict[str, str]:
    return {oid: f"P{i:03d}" for i, oid in enumerate(original_ids, start=1)}


# =============================================================================
# Units (px/mm)
# =============================================================================
def px_to_mm(length_px: float, px_per_mm: float) -> float:
    px_per_mm = float(max(px_per_mm, 1e-12))
    return float(length_px) / px_per_mm


def px2_to_mm2(area_px2: float, px_per_mm: float) -> float:
    px_per_mm = float(max(px_per_mm, 1e-12))
    return float(area_px2) / (px_per_mm ** 2)


# =============================================================================
# Geometry helpers
# =============================================================================
def _ensure_closed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) >= 3 and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return x, y


def OutlinePerimeter(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))


def OutlineArea(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _ensure_closed(x, y)
    if len(x) < 4:
        return 0.0
    return float(0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def OutlineCentroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(x)), float(np.mean(y))


def ellipse_perimeter(a: float, b: float) -> float:
    """Ellipse perimeter (Carlson via SciPy if available, else Ramanujan)."""
    a = float(abs(a))
    b = float(abs(b))
    if a <= 0 or b <= 0:
        return 0.0

    if not SCIPY_SPECIAL_AVAILABLE:
        h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
        return float(math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(max(1e-12, 4 - 3 * h)))))

    if b > a:
        a, b = b, a
    e2 = 1.0 - (b / a) ** 2
    m = float(np.clip(e2, 0.0, 1.0))
    RF = elliprf(0.0, 1.0 - m, 1.0)
    RD = elliprd(0.0, 1.0 - m, 1.0)
    E = RF - (m / 3.0) * RD
    return float(4.0 * a * E)


def equal_area_circle_radius(x: np.ndarray, y: np.ndarray) -> float:
    area = OutlineArea(x, y)
    return float(math.sqrt(area / math.pi)) if area > 0 else 0.0


def fit_ellipse_equal_area(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Covariance-based ellipse orientation + scale to match polygon area.
    Returns (cx, cy, a, b, theta) with a>=b.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = OutlineCentroid(x, y)
    X = np.stack([x - cx, y - cy], axis=1)

    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    area = OutlineArea(x, y)
    if area <= 0:
        area = float(np.pi * max(vals[0], 1e-12) * max(vals[1], 1e-12))

    a0 = math.sqrt(max(vals[0], 1e-12))
    b0 = math.sqrt(max(vals[1], 1e-12))
    scale = math.sqrt(area / (math.pi * a0 * b0 + 1e-12))

    a = a0 * scale
    b = b0 * scale
    theta = math.atan2(vecs[1, 0], vecs[0, 0])

    if b > a:
        a, b = b, a
        theta += math.pi / 2.0

    return cx, cy, a, b, theta


def ShapeFactors(area: float, perimeter: float) -> Tuple[float, float, float, float]:
    """(Circularity, Sphericity, Bulkiness, Surface_roughness)."""
    if area <= 0 or perimeter <= 0:
        return (float("nan"),) * 4
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter + 1e-12))
    sphericity = float(math.sqrt(max(circularity, 0.0)))
    bulkiness = float((perimeter * perimeter) / (4.0 * math.pi * area + 1e-12))

    r = math.sqrt(area / math.pi)
    circ_p = 2.0 * math.pi * r
    roughness = float(perimeter / (circ_p + 1e-12))
    return circularity, sphericity, bulkiness, roughness


# =============================================================================
# Elliptic Fourier (simple)
# =============================================================================
def ComputeEllFourierCoef(x: np.ndarray, y: np.ndarray, n_harmonics: int):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = _ensure_closed(x, y)

    Ax = np.zeros(n_harmonics + 1, dtype=np.float64)
    Bx = np.zeros(n_harmonics + 1, dtype=np.float64)
    Ay = np.zeros(n_harmonics + 1, dtype=np.float64)
    By = np.zeros(n_harmonics + 1, dtype=np.float64)

    if len(x) < 4:
        Ax[0] = x[0] if len(x) else 0.0
        Ay[0] = y[0] if len(y) else 0.0
        return Ax, Bx, Ay, By

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.hypot(dx, dy)
    dt[dt == 0] = 1e-12
    t = np.concatenate(([0.0], np.cumsum(dt)))
    T = float(t[-1]) if float(t[-1]) > 0 else 1.0

    for n in range(1, n_harmonics + 1):
        cn = 2.0 * math.pi * n / T
        cos_ct = np.cos(cn * t)
        sin_ct = np.sin(cn * t)

        Ax[n] = (1.0 / (cn * cn * T)) * np.sum((dx / dt) * (cos_ct[1:] - cos_ct[:-1]))
        Bx[n] = (1.0 / (cn * cn * T)) * np.sum((dx / dt) * (sin_ct[1:] - sin_ct[:-1]))
        Ay[n] = (1.0 / (cn * cn * T)) * np.sum((dy / dt) * (cos_ct[1:] - cos_ct[:-1]))
        By[n] = (1.0 / (cn * cn * T)) * np.sum((dy / dt) * (sin_ct[1:] - sin_ct[:-1]))

    Ax[0] = float(np.mean(x))
    Ay[0] = float(np.mean(y))
    return Ax, Bx, Ay, By


# =============================================================================
# Contour-fit metrics (RMSE intentionally omitted)
# =============================================================================
def _resample_closed_contour(x: np.ndarray, y: np.ndarray, m: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _ensure_closed(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(x) < 4:
        return x, y

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12
    s = np.concatenate(([0.0], np.cumsum(ds)))
    total = float(s[-1]) if float(s[-1]) > 0 else 1.0

    u = np.linspace(0.0, total, m, endpoint=False)
    xr = np.interp(u, s, x)
    yr = np.interp(u, s, y)
    return xr, yr


def _best_shift_sse(A: np.ndarray, B: np.ndarray) -> float:
    m = A.shape[0]
    best = np.inf
    for k in range(m):
        sse = float(np.sum((A - np.roll(B, k, axis=0)) ** 2))
        if sse < best:
            best = sse
    return best


def contour_fit_metrics(x, y, xt, yt, m: int = 200) -> Tuple[float, float]:
    """Return (R², NRMSE). RMSE is not computed or displayed by design."""
    x1, y1 = _resample_closed_contour(x, y, m=m)
    x2, y2 = _resample_closed_contour(xt, yt, m=m)

    A = np.stack([x1, y1], axis=1)
    B = np.stack([x2, y2], axis=1)

    sse = min(_best_shift_sse(A, B), _best_shift_sse(A, B[::-1]))
    sst = float(np.sum((A - A.mean(axis=0)) ** 2))

    r2 = 1.0 - sse / max(sst, 1e-12)
    nrmse = float(math.sqrt(sse / max(sst, 1e-12)))
    return r2, nrmse


# =============================================================================
# Fixed computation params
# =============================================================================
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


PARAMS_FIXED = ShapeParams(
    n_harmonics=16,
    no_sum=16,
    large_limit=100.0,
    small_limit=1e-5,
    flag_location=1,
    flag_scale=1,
    flag_rotation=1,
    flag_start=1,
)


# =============================================================================
# Batch computation
# =============================================================================
def compute_shape_indices(df_xy: pd.DataFrame, params: ShapeParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    errors = []

    no_sum = int(max(1, min(params.no_sum, params.n_harmonics)))

    for pid, g in df_xy.groupby("ID"):
        try:
            x = g["X"].to_numpy(dtype=float)
            y = g["Y"].to_numpy(dtype=float)
            if len(x) < 3:
                raise ValueError("Too few points")

            x, y = _ensure_closed(x, y)

            area = OutlineArea(x, y)
            per = OutlinePerimeter(x, y)

            cx, cy, a, b, theta = fit_ellipse_equal_area(x, y)
            ell_per = ellipse_perimeter(a, b)

            k = float(a / b) if b > 0 else float("nan")
            elongation = float(1.0 - (b / a)) if a > 0 else float("nan")

            Ax, Bx, Ay, By = ComputeEllFourierCoef(x, y, int(params.n_harmonics))

            eps = 1e-12
            E = np.array(
                [float(Ax[n] ** 2 + Bx[n] ** 2 + Ay[n] ** 2 + By[n] ** 2) for n in range(1, no_sum + 1)],
                dtype=float,
            )
            E_total = float(np.sum(E) + eps)

            # Polygonality: energy beyond the first harmonic (classic proxy)
            P = float(np.sum(E[1:]) / E_total) if len(E) >= 2 else 0.0

            Ex = float(np.sum([Ax[n] ** 2 + Bx[n] ** 2 for n in range(1, no_sum + 1)]) + eps)
            Ey = float(np.sum([Ay[n] ** 2 + By[n] ** 2 for n in range(1, no_sum + 1)]) + eps)

            Ass = float(abs(Ex - Ey) / (Ex + Ey))  # Assymmetricity
            Ass_norm = float(Ass / max(P, eps))

            angularity = float(np.clip((per / max(ell_per, eps)) - 1.0, 0.0, params.large_limit))

            circularity, sphericity, bulkiness, roughness = ShapeFactors(area, per)

            results.append(
                {
                    "Original_ID": pid,
                    "Elongation_ratio": k,
                    "Elongation": elongation,
                    "Angularity": angularity,
                    "Surface_roughness": roughness,
                    "Asymmetricity": Ass,
                    "Asymmetricity_normalized": Ass_norm,
                    "Polygonality": P,
                    "Area": area,
                    "Perimeter": per,
                    "Bulkiness": bulkiness,
                    "Circularity": circularity,
                    "Sphericity": sphericity,
                }
            )

        except Exception as e:
            errors.append({"ID": pid, "error": str(e)})

    return pd.DataFrame(results), pd.DataFrame(errors)


# =============================================================================
# Module 1 plots
# =============================================================================
def plot_reconstruction_spatial_efd(x: np.ndarray, y: np.ndarray, K: int):
    x, y = _ensure_closed(x, y)
    coeffs = spatial_efd.CalculateEFD(x, y, K)
    locus = spatial_efd.calculate_dc_coefficients(x, y)
    xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=K, locus=locus)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=170)
    ax.plot(x, y, linewidth=1.3, label="original outline")
    ax.plot(xt, yt, linewidth=1.3, label=f"Reconstructed (K={K})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Outline reconstruction", fontsize=11)
    return fig, xt, yt


def plot_ellipse_and_circle(x: np.ndarray, y: np.ndarray):
    cx, cy, a, b, theta = fit_ellipse_equal_area(x, y)
    r = equal_area_circle_radius(x, y)

    t = np.linspace(0, 2 * np.pi, 300)
    ex = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ey = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    cxr = cx + r * np.cos(t)
    cyr = cy + r * np.sin(t)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=170)
    ax.plot(x, y, linewidth=1.3, label="original outline")
    ax.plot(ex, ey, linewidth=1.3, label="ellipse")
    ax.plot(cxr, cyr, linewidth=1.3, label="Circle")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Equla-area & Ellipse of circle", fontsize=11)
    return fig


# =============================================================================
# Figure_5-style histogram helper (percentiles legend)
# =============================================================================
def percentile_histogram(df: pd.DataFrame, col: str, xlabel: str) -> plt.Figure:
    data = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=170)

    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    p10 = float(np.percentile(data, 10))
    p50 = float(np.percentile(data, 50))
    p90 = float(np.percentile(data, 90))

    ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.legend([f"10% = {p10:.3f}\n50% = {p50:.3f}\n90% = {p90:.3f}"], fontsize=11, frameon=True)
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    return fig


def general_statistics_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    desc = numeric.describe(percentiles=[0.10, 0.50, 0.90]).T
    desc = desc.rename(
        columns={
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "10%": "p10",
            "50%": "p50",
            "90%": "p90",
            "max": "max",
        }
    )
    return desc.reset_index().rename(columns={"index": "descriptor"})


# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Shape Analysis", layout="wide")
st.title("🔬 Shape Analysis")

uploaded = st.sidebar.file_uploader("Upload CSV (ID, X, Y)", type=["csv"])

# Left part: removed compact mode, kept text size
font_size = st.sidebar.slider("Text size", min_value=14, max_value=24, value=18, step=1)
st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{ font-size: {font_size}px !important; }}
      section[data-testid="stSidebar"] * {{ font-size: {font_size}px !important; }}
      .stMetricValue {{ font-size: {min(font_size+12, 34)}px !important; }}
      .stMetricLabel {{ font-size: {max(font_size-2, 12)}px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

EXPANDED = True

st.sidebar.markdown("---")
module = st.sidebar.radio("Modules", ["Module 1", "Module 2", "Module 3"], index=0)

st.sidebar.markdown("---")
# Renamed Units Calibration -> Set Scale; label px/mm; removed captions
with st.sidebar.expander("📏 Set Scale", expanded=True):
    px_per_mm = st.number_input("px/mm", min_value=0.0001, value=77.0, step=0.1)

if uploaded is None:
    st.info("⬅️ Upload a CSV to start.")
    st.caption("Expected columns: ID, X, Y.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _load_cached(h: str, b: bytes) -> pd.DataFrame:
    return load_xy_csv(b)


df_xy = _load_cached(file_hash, file_bytes)
if df_xy.empty:
    st.error("CSV loaded but has no valid rows for ID/X/Y.")
    st.stop()

original_ids = df_xy["ID"].dropna().astype(str).unique().tolist()
id_map = make_id_mapping(original_ids)
reverse_map = {v: k for k, v in id_map.items()}

st.caption(f"Detected **{df_xy['ID'].nunique()}** particles and **{len(df_xy)}** outline points.")


# =============================================================================
# Module 1
# =============================================================================
if module == "Module 1":
    st.header("Module 1 — Outline Reconstruction")

    # RMSE removed from instructions and not computed anywhere in Module 1
    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Choose **one** particle and harmonic orders from **1 to 40**. "
            "Observe how the outline converges to the original shape."
        )

    # Renamed figure section
    show_images(["Module1_1.png", "Module1_2.png"], "📌 EFA method", expanded=EXPANDED, ncols=1)

    display_ids = [id_map[o] for o in original_ids]
    chosen_display = st.selectbox("Select ID", display_ids, index=0)
    chosen_original = reverse_map.get(chosen_display, chosen_display)

    g = df_xy[df_xy["ID"].astype(str) == str(chosen_original)]
    x = g["X"].to_numpy(dtype=float)
    y = g["Y"].to_numpy(dtype=float)

    if len(x) < 3:
        st.error("This particle has <3 points.")
        st.stop()

    area_px2 = float(OutlineArea(x, y))
    per_px = float(OutlinePerimeter(x, y))

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        st.metric("Area (px²)", f"{area_px2:.4g}")
        st.metric("Area (mm²)", f"{px2_to_mm2(area_px2, px_per_mm):.4g}")
    with c2:
        st.metric("Perimeter (px)", f"{per_px:.4g}")
        st.metric("Perimeter (mm)", f"{px_to_mm(per_px, px_per_mm):.4g}")
    with c3:
        st.metric("px/mm", f"{px_per_mm:.4g}")

    if not SPATIAL_EFD_AVAILABLE:
        st.warning("Module 1 reconstruction needs `spatial_efd` (pip install spatial_efd).")
        st.stop()

    nyq = int(spatial_efd.Nyquist(x))
    maxK = max(1, min(40, nyq))
    K = st.slider("Harmonic order (K)", min_value=1, max_value=maxK, value=min(10, maxK))
    st.caption(f"Nyquist limit: **{nyq}** (K slider capped at **{maxK}**)")

    try:
        fig_rec, xt, yt = plot_reconstruction_spatial_efd(x, y, int(K))
        r2, nrmse = contour_fit_metrics(x, y, xt, yt, m=200)

        m1, m2 = st.columns(2)
        m1.metric("Reconstruction Quality (R²)", f"{r2:.3f}")
        m2.metric("NRMSE", f"{nrmse:.4g}")
    except Exception as e:
        st.warning(f"Reconstruction metric failed: {e}")
        fig_rec = None

    # Renamed "Figures" -> "Reconstruction of selected particle"
    with st.expander("Reconstruction of selected particle", expanded=EXPANDED):
        left, right = st.columns(2)
        with left:
            if fig_rec is not None:
                st.pyplot(fig_rec, use_container_width=True)
        with right:
            st.pyplot(plot_ellipse_and_circle(x, y), use_container_width=True)


# =============================================================================
# Module 2
# =============================================================================
elif module == "Module 2":
    st.header("Module 2 — Sensitivity Analysis")

    # Instruction: delete 2nd sentence
    with st.expander("📘 Instructions", expanded=True):
        st.write("Apply different harmonic orders to the whole dataset.")

    # Renamed "Module 2 instructions figures"
    show_images(["Module2_1.png"], "📌 Asymmetricity and Polygonality equations", expanded=EXPANDED, ncols=1)

    max_h = 40
    default_N = [1, 2, 5, 10, 15, 20]
    chosen_N = st.multiselect(
        "Select harmonics order to evaluate",
        options=list(range(1, max_h + 1)),
        default=[n for n in default_N if n <= max_h],
    )
    chosen_N = sorted(set(int(n) for n in chosen_N))
    if not chosen_N:
        st.info("Choose at least one harmonic order to evaluate.")
        st.stop()

    current_key = (file_hash, tuple(chosen_N))

    if st.session_state.get("module2_key") == current_key:
        sens = st.session_state.get("module2_sens")
    else:
        sens = None
        st.info("Settings changed. Click **Run sensitivity** to compute.")

    @st.cache_data(show_spinner=False)
    def _compute_for_N(h: str, df_xy_local: pd.DataFrame, N: int) -> pd.DataFrame:
        p = ShapeParams(
            n_harmonics=int(N),
            no_sum=int(N),
            large_limit=PARAMS_FIXED.large_limit,
            small_limit=PARAMS_FIXED.small_limit,
            flag_location=PARAMS_FIXED.flag_location,
            flag_scale=PARAMS_FIXED.flag_scale,
            flag_rotation=PARAMS_FIXED.flag_rotation,
            flag_start=PARAMS_FIXED.flag_start,
        )
        res, _ = compute_shape_indices(df_xy_local, p)
        if res.empty:
            return pd.DataFrame()
        out = res[["Original_ID", "Asymmetricity", "Polygonality"]].copy()
        out["Harmonics"] = int(N)
        return out

    run = st.button("Run sensitivity", type="primary")
    if run:
        prog = st.progress(0)
        status = st.empty()

        rows = []
        total = len(chosen_N)
        for i, N in enumerate(chosen_N, start=1):
            status.write(f"Computing N = {N}  ({i}/{total})")
            part = _compute_for_N(file_hash, df_xy, int(N))
            if not part.empty:
                rows.append(part)
            prog.progress(int(i / max(total, 1) * 100))

        status.empty()
        prog.empty()

        if not rows:
            st.error("No sensitivity data produced.")
            st.stop()

        sens = pd.concat(rows, ignore_index=True)
        sens["Particle_ID"] = sens["Original_ID"].map(id_map).fillna(sens["Original_ID"])

        st.session_state["module2_sens"] = sens
        st.session_state["module2_key"] = current_key

    if sens is None:
        st.stop()

    sens_sel = sens[sens["Harmonics"].isin(chosen_N)].copy()
    sens_sel["Harmonics_cat"] = sens_sel["Harmonics"].astype(str)
    cat_order = [str(n) for n in chosen_N]

    # Keep just "Boxplots"
    with st.expander("📦 Boxplots", expanded=EXPANDED):
        figA = px.box(
            sens_sel,
            x="Harmonics_cat",
            y="Asymmetricity",
            points="outliers",
            height=320,
            title="Asymmetricity vs harmonics",
        )
        figA.update_xaxes(type="category", categoryorder="array", categoryarray=cat_order, title="Number of harmonics")
        figA.update_layout(yaxis_title="Asymmetricity")
        st.plotly_chart(figA, use_container_width=True)

        figP = px.box(
            sens_sel,
            x="Harmonics_cat",
            y="Polygonality",
            points="outliers",
            height=320,
            title="Polygonality vs harmonics",
        )
        figP.update_xaxes(type="category", categoryorder="array", categoryarray=cat_order, title="Number of harmonics")
        figP.update_layout(yaxis_title="Polygonality")
        st.plotly_chart(figP, use_container_width=True)

    # "Stabilization trend bt medians"
    with st.expander("📉 Stabilization trend bt medians", expanded=False):
        trend = sens.groupby("Harmonics")[["Asymmetricity", "Polygonality"]].median().reset_index()
        cA, cP = st.columns(2)
        with cA:
            st.plotly_chart(
                px.line(trend, x="Harmonics", y="Asymmetricity", markers=True, height=300,
                        title="Median Asymmetricity vs harmonics"),
                use_container_width=True,
            )
        with cP:
            st.plotly_chart(
                px.line(trend, x="Harmonics", y="Polygonality", markers=True, height=300,
                        title="Median Polygonality vs harmonics"),
                use_container_width=True,
            )

    with st.expander("🧾 Sensitivity table", expanded=EXPANDED):
        st.dataframe(
            sens_sel[["Particle_ID", "Harmonics", "Asymmetricity", "Polygonality"]]
            .sort_values(["Harmonics", "Particle_ID"]),
            use_container_width=True,
        )

    # Renamed download button text
    st.download_button(
        "⬇️ Download sensitivity table",
        data=sens_sel[["Particle_ID", "Harmonics", "Asymmetricity", "Polygonality"]].to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_table.csv",
        mime="text/csv",
    )


# =============================================================================
# Module 3
# =============================================================================
else:
    st.header("Module 3 — Statistics")

    with st.expander("📘 Instructions", expanded=True):
        st.write(
            "Run, plot distributions (ELongation, Angularity, Roughness, Asymmetricity and Polygonality) "
            "and correlation matrix to find redundant descriptors"
        )

    show_images(
        ["Module3_1.png", "Module3_2.png", "Module3_3.png", "Module3_4.png"],
        "📌 Module 3 instructions (figures)",
        expanded=EXPANDED,
        ncols=2,
    )

    key3 = (file_hash, float(px_per_mm))

    if st.session_state.get("module3_key") == key3:
        results_df = st.session_state.get("module3_results")
        errors_df = st.session_state.get("module3_errors")
    else:
        results_df = None
        errors_df = None
        st.info("Settings changed. Click **EFA to whole dataset** to compute.")

    # Renamed run button
    run3 = st.button("EFA to whole dataset", type="primary")
    if run3:
        res, err = compute_shape_indices(df_xy, PARAMS_FIXED)
        if res.empty:
            st.error("No results produced.")
            st.stop()

        res = res.copy()
        res["Particle_ID"] = res["Original_ID"].map(id_map).fillna(res["Original_ID"])

        res["Area_mm2"] = res["Area"].apply(lambda v: px2_to_mm2(v, px_per_mm))
        res["Perimeter_mm"] = res["Perimeter"].apply(lambda v: px_to_mm(v, px_per_mm))

        st.session_state["module3_results"] = res
        st.session_state["module3_errors"] = err
        st.session_state["module3_key"] = key3

        results_df = res
        errors_df = err

    if results_df is None:
        st.stop()

    # Added "Results - all shape indices"
    st.subheader("Results - all shape indices")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "⬇️ Download full results (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="shape_indices_full.csv",
        mime="text/csv",
    )

    if errors_df is not None and not errors_df.empty:
        with st.expander("Errors", expanded=False):
            st.dataframe(errors_df, use_container_width=True)

    # Added table of General statistics
    with st.expander("📋 General statistics", expanded=EXPANDED):
        stat_cols = ["Elongation", "Angularity", "Surface_roughness", "Asymmetricity", "Polygonality"]
        stat_cols = [c for c in stat_cols if c in results_df.columns]
        stats_tbl = general_statistics_table(results_df, stat_cols)
        st.dataframe(stats_tbl, use_container_width=True)

        st.download_button(
            "⬇️ Download general statistics (CSV)",
            data=stats_tbl.to_csv(index=False).encode("utf-8"),
            file_name="general_statistics.csv",
            mime="text/csv",
        )

    # Distributions like Figure_5.ipynb (percentiles legend)
    with st.expander("📊 Distributions (ELongation, Angularity, Roughness, Asymmetricity, Polygonality)", expanded=EXPANDED):
        dist_cols = [
            ("Elongation", "Elongation"),
            ("Angularity", "Angularity"),
            ("Surface_roughness", "Surface roughness"),
            ("Asymmetricity", "Asymmetricity"),
            ("Polygonality", "Polygonality"),
        ]
        for col, xlabel in dist_cols:
            if col not in results_df.columns:
                continue
            st.pyplot(percentile_histogram(results_df, col, xlabel), use_container_width=False)

    # Correlation matrix: warm-ish + values in boxes
    with st.expander("Correlation matrix", expanded=EXPANDED):
        corr_cols = ["Elongation", "Angularity", "Surface_roughness", "Asymmetricity", "Polygonality"]
        corr_cols = [c for c in corr_cols if c in results_df.columns]
        corr = results_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

        st.download_button(
            "⬇️ Download correlation matrix (CSV)",
            data=corr.to_csv().encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

        fig_corr = px.imshow(
            corr,
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",  # warm for positive, cool for negative
            text_auto=".2f",
            title="Correlation (Pearson)",
        )
        fig_corr.update_xaxes(side="bottom")
        st.plotly_chart(fig_corr, use_container_width=True)
