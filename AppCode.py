import streamlit as st
import pandas as pd
import os
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Try to enable KDE; if scipy is not available, we'll just show histograms
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Particle Comparison Viewer", layout="wide")

st.title("🔬 Particle Comparison Viewer")

# ============================
# SIDEBAR: FILE UPLOADS
# ============================
st.sidebar.header("1. Upload your data")

uploaded_csv = st.sidebar.file_uploader(
    "Upload particle CSV file", type=["csv"], key="csv_uploader"
)

uploaded_images = st.sidebar.file_uploader(
    "Upload particle images (select multiple files)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="img_uploader",
)

if uploaded_csv is None:
    st.warning("⬅️ Please upload a CSV file in the sidebar to continue.")
    st.stop()

if not uploaded_images:
    st.warning("⬅️ Please upload particle images (multiple files) in the sidebar to continue.")
    st.stop()

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

df = load_data(uploaded_csv.getvalue())

if "ID" not in df.columns:
    st.error("The uploaded CSV must contain an 'ID' column (matching image filenames).")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ============================
# IMAGE DICTIONARY
# ============================
def build_image_dict(files):
    img_dict = {}
    for f in files:
        stem, _ = os.path.splitext(f.name)  # "123.png" -> "123"
        try:
            img = Image.open(f).copy()
            img_dict[stem] = img
        except Exception:
            # Skip unreadable images
            continue
    return img_dict

if "image_dict" not in st.session_state:
    st.session_state["image_dict"] = build_image_dict(uploaded_images)
    st.session_state["image_names"] = [f.name for f in uploaded_images]
else:
    current_names = [f.name for f in uploaded_images]
    if st.session_state.get("image_names") != current_names:
        st.session_state["image_dict"] = build_image_dict(uploaded_images)
        st.session_state["image_names"] = current_names

image_dict = st.session_state["image_dict"]

# ============================
# SIDEBAR: CONTROLS
# ============================
st.sidebar.title("2. Particle Viewer Controls")

particle_ids = df["ID"].tolist()
particle1 = st.sidebar.selectbox("Select Particle 1", particle_ids, index=0)
particle2 = st.sidebar.selectbox(
    "Select Particle 2", particle_ids, index=min(1, len(particle_ids) - 1)
)

# Optional filter
df_filtered = df.copy()
with st.sidebar.expander("🔍 Filter Dataset"):
    if numeric_cols:
        filter_col = st.selectbox("Select column to filter", options=numeric_cols)
        min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
        val_range = st.slider("Value range", min_val, max_val, (min_val, max_val))
        df_filtered = df[
            (df[filter_col] >= val_range[0]) & (df[filter_col] <= val_range[1])
        ]
    else:
        st.info("No numeric columns found to filter.")

# ============================
# TABS
# ============================
tab1, tab2, tab3 = st.tabs(
    ["🧾 Compare Particles", "📈 Explore Dataset", "📊 Analytics"]
)

# ============================
# HELPER: DISPLAY PARTICLE
# ============================
def display_particle(particle_id, df, img_dict):
    st.subheader(f"Particle {particle_id}")

    key = str(particle_id)
    img = img_dict.get(key, None)

    if img is not None:
        st.image(img, caption=f"Image: {particle_id}", use_container_width=True)
    else:
        st.error("❌ No image found with a matching filename")

    row = df[df["ID"] == particle_id]
    st.dataframe(row, use_container_width=True)

# ============================
# TAB 1: COMPARISON
# ============================
with tab1:
    st.header("🧾 Particle Comparison")

    col1, col2 = st.columns(2)
    with col1:
        display_particle(particle1, df, image_dict)
    with col2:
        display_particle(particle2, df, image_dict)

    # Comparison bar chart
    if numeric_cols:
        compare_cols = st.multiselect(
            "Select properties to compare",
            numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
        )
        if compare_cols:
            compare_df = df[df["ID"].isin([particle1, particle2])][["ID"] + compare_cols]
            compare_df_melted = compare_df.melt(
                id_vars="ID", var_name="Property", value_name="Value"
            )
            fig = px.bar(
                compare_df_melted,
                x="Property",
                y="Value",
                color="ID",
                barmode="group",
                title="Particle Property Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available for comparison plots.")

# ============================
# TAB 2: DATASET EXPLORER
# ============================
with tab2:
    st.header("📈 Dataset Explorer")

    st.dataframe(df_filtered, use_container_width=True)

    if len(numeric_cols) >= 2:
        colX, colY = st.columns(2)
        with colX:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
        with colY:
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)

        fig2 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            hover_name="ID",
            title="Scatter Plot of Particles",
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif len(numeric_cols) == 1:
        st.info("Only one numeric column found, scatter plot requires at least two.")
    else:
        st.info("No numeric columns available for scatter plot.")

    # Download filtered data
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV", csv, "filtered_particles.csv", "text/csv"
    )

# ============================
# TAB 3: ANALYTICS
# ============================
with tab3:
    st.header("📊 Analytics & Statistics")

    if not numeric_cols:
        st.info("No numeric columns found for analytics.")
    else:
        # -------- General & extra statistics --------
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.subheader("Summary statistics")
            desc = df[numeric_cols].describe().T
            st.dataframe(desc, use_container_width=True)

        with col_stat2:
            st.subheader("Extra statistics")
            extra_stats = pd.DataFrame(
                {
                    "skew": df[numeric_cols].skew(),
                    "kurtosis": df[numeric_cols].kurt(),
                    "missing_count": df[numeric_cols].isna().sum(),
                    "missing_pct": df[numeric_cols].isna().mean() * 100,
                }
            )
            st.dataframe(extra_stats, use_container_width=True)

        st.markdown("---")

        # -------- Boxplots --------
        st.subheader("Box plots")
        box_cols = st.multiselect(
            "Select columns for box plots",
            numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
        )

        if box_cols:
            for col in box_cols:
                fig_box = px.box(
                    df,
                    y=col,
                    points="outliers",
                    title=f"Boxplot: {col}",
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Select at least one column to see its box plot.")

        st.markdown("---")

        # -------- Distribution + KDE --------
        st.subheader("Distribution + KDE")

        dist_col = st.selectbox(
            "Select column for distribution", numeric_cols, index=0
        )
        data = df[dist_col].dropna().values  # 👈 this line is correctly indented

        if data.size > 0:
            fig_dist = go.Figure()
            # Histogram (density)
            fig_dist.add_trace(
                go.Histogram(
                    x=data,
                    histnorm="probability density",
                    nbinsx=30,
                    name="Histogram",
                    opacity=0.6,
                )
            )

            # KDE (if scipy available)
            if SCIPY_AVAILABLE and data.size > 1:
                xs = np.linspace(data.min(), data.max(), 400)
                kde = gaussian_kde(data)
                fig_dist.add_trace(
                    go.Scatter(
                        x=xs,
                        y=kde(xs),
                        mode="lines",
                        name="KDE",
                    )
                )

            fig_dist.update_layout(
                title=f"Distribution + KDE: {dist_col}",
                xaxis_title=dist_col,
                yaxis_title="Density",
                barmode="overlay",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No data available for selected column.")

        st.markdown("---")

        # -------- Correlation matrix --------
        st.subheader("Correlation matrix (Pearson)")
        corr = df[numeric_cols].corr(method="pearson")
        st.dataframe(corr, use_container_width=True)

        fig_corr = px.imshow(
            corr,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation matrix (Pearson)",
        )
        fig_corr.update_xaxes(side="bottom")
        st.plotly_chart(fig_corr, use_container_width=True)
