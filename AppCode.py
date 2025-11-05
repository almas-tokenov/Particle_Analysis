import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px

# --- CONFIG ---
IMAGE_FOLDER = "Images"  # folder where particle images are stored
DATA_FILE = "particles_fixed.csv"  # use your uploaded file

st.set_page_config(page_title="Particle Comparison Viewer", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🔬 Particle Viewer Controls")
particle_ids = df["ID"].tolist()
particle1 = st.sidebar.selectbox("Select Particle 1", particle_ids, index=0)
particle2 = st.sidebar.selectbox("Select Particle 2", particle_ids, index=1)

# Optional filters
with st.sidebar.expander("🔍 Filter Dataset"):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    filter_col = st.selectbox("Select column to filter", options=numeric_cols)
    min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
    val_range = st.slider("Value range", min_val, max_val, (min_val, max_val))
    df_filtered = df[(df[filter_col] >= val_range[0]) & (df[filter_col] <= val_range[1])]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🧾 Compare Particles", "📈 Explore Dataset", "📊 Statistics"])

# --- DISPLAY FUNCTION ---
def display_particle(particle_id):
    st.subheader(f"Particle {particle_id}")
    image_path = os.path.join(IMAGE_FOLDER, f"{particle_id}.png")

    # Image
    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption=f"Image: {particle_id}", use_container_width=True)
    else:
        st.error("❌ No image found")

    # Data
    row = df[df["ID"] == particle_id]
    st.dataframe(row, use_container_width=True)

# --- TAB 1: COMPARISON ---
with tab1:
    st.header("🧾 Particle Comparison")

    col1, col2 = st.columns(2)
    with col1:
        display_particle(particle1)
    with col2:
        display_particle(particle2)

    # Comparison chart
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    compare_cols = st.multiselect("Select properties to compare", numeric_cols, numeric_cols[:3])
    if compare_cols:
        compare_df = df[df["ID"].isin([particle1, particle2])][["ID"] + compare_cols]
        compare_df_melted = compare_df.melt(id_vars="ID", var_name="Property", value_name="Value")
        fig = px.bar(compare_df_melted, x="Property", y="Value", color="ID", barmode="group",
                     title="Particle Property Comparison")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: EXPLORE DATASET ---
with tab2:
    st.header("📈 Dataset Explorer")

    st.dataframe(df_filtered, use_container_width=True)
    colX, colY = st.columns(2)
    with colX:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
    with colY:
        y_col = st.selectbox("Y-axis", numeric_cols, index=1)

    fig2 = px.scatter(df_filtered, x=x_col, y=y_col, hover_name="ID", title="Scatter Plot of Particles")
    st.plotly_chart(fig2, use_container_width=True)

    # Download filtered data
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download filtered CSV", csv, "filtered_particles.csv", "text/csv")

# --- TAB 3: STATISTICS ---
with tab3:
    st.header("📊 Summary Statistics")
    st.write(df.describe())

    fig3 = px.histogram(df_filtered, x=filter_col, nbins=30, title=f"Distribution of {filter_col}")
    st.plotly_chart(fig3, use_container_width=True)