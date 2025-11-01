import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="🏎️ F1 Driver Cluster Visualizer", layout="wide")
st.title("🏁 Formula 1 Driver Clustering (K-Means)")
st.write("Group Formula 1 drivers by **performance efficiency metrics** using K-Means clustering.")

data = pd.read_csv("F1DriversDataset.csv")
st.subheader("Dataset Preview")
st.dataframe(data.head())

features = ["Win_Rate", "Podium_Rate", "FastLap_Rate", "Pole_Rate", "Points_Per_Entry"]
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters (k):", 2, 8, 4)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)

def describe_cluster(row):
    desc = []
    if row["Win_Rate"] > 0.25: desc.append("frequent winners")
    elif row["Win_Rate"] > 0.05: desc.append("occasional winners")
    else: desc.append("rare winners")
    if row["Podium_Rate"] > 0.3: desc.append("consistent podiums")
    elif row["Podium_Rate"] > 0.1: desc.append("some podiums")
    if row["Points_Per_Entry"] > 5: desc.append("high points per race")
    elif row["Points_Per_Entry"] > 2: desc.append("moderate scoring")
    if row["FastLap_Rate"] > 0.3: desc.append("often fastest laps")
    return ", ".join(desc).capitalize() or "Developing drivers"

summaries = []
for idx, row in centers.iterrows():
    summaries.append(f"**Cluster {idx}** — {describe_cluster(row)}.")

st.subheader("Cluster Centers (Average Performance per Cluster)")
st.dataframe(centers.style.background_gradient(cmap="YlOrRd"))

norm_centers = (centers - centers.min()) / (centers.max() - centers.min())
fig_radar = go.Figure()
for i in range(k):
    fig_radar.add_trace(go.Scatterpolar(
        r=norm_centers.iloc[i].values,
        theta=features,
        fill='toself',
        name=f'Cluster {i}'
    ))
fig_radar.update_layout(
    title="Cluster Profile Comparison (Normalized)",
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    showlegend=True
)
st.plotly_chart(fig_radar, use_container_width=True)


st.subheader("2-D Cluster Visualization")

x_axis = st.selectbox("X-axis feature (2D):", features, index=0, key="x2d")
y_axis = st.selectbox("Y-axis feature (2D):", features, index=2, key="y2d")

fig_2d = px.scatter(
    data,
    x=x_axis,
    y=y_axis,
    color=data["Cluster"].astype(str),
    hover_name="Driver",
    text="Driver",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title="Driver Distribution by Cluster (2D)"
)
fig_2d.update_traces(
    textposition="top center",
    marker=dict(size=9, line=dict(width=0.5, color='white'))
)
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("**Color Legend:**")
st.markdown("""
- 🟣 **Cluster 0** – Top-tier champions (high win and podium rates)  
- 🟠 **Cluster 2** – Consistent performers (occasional wins and podiums)  
- 🟢 **Cluster 1** – Developing or low-performing drivers (few points, low rates)  
- 🔵 **Cluster 3** – Retired or rare participants (minimal activity)
""")

st.subheader("3-D Cluster Visualization 🌐")
x3 = st.selectbox("X-axis feature (3D):", features, index=0, key="x3d")
y3 = st.selectbox("Y-axis feature (3D):", features, index=1, key="y3d")
z3 = st.selectbox("Z-axis feature (3D):", features, index=2, key="z3d")

fig_3d = px.scatter_3d(
    data,
    x=x3, y=y3, z=z3,
    color=data["Cluster"].astype(str),
    hover_name="Driver",
    hover_data=features,
    title="Driver Clusters in 3D Space",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_3d.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown("**Color Legend:**")
st.markdown("""
- 🟣 **Cluster 0** – Top-tier champions (high win and podium rates)  
- 🟠 **Cluster 2** – Consistent performers (occasional wins and podiums)  
- 🟢 **Cluster 1** – Developing or low-performing drivers (few points, low rates)  
- 🔵 **Cluster 3** – Retired or rare participants (minimal activity)
""")

st.subheader("Cluster Insights 🧠")
for summary in summaries:
    st.markdown(f"- {summary}")

st.subheader("Drivers in Each Cluster")
driver_groups = data.groupby("Cluster")["Driver"].apply(lambda x: ', '.join(x)).reset_index()
driver_groups = driver_groups.rename(columns={"Driver": "Drivers"})
st.dataframe(driver_groups)

st.markdown("---")
st.caption("📘 Murmani Akhaladze — Numerical Programming Fall 2025")

# ======================
#   HOW TO RUN
# ======================
# 1. Save this file as: ap2_frontend.py
# 2. Place F1DriversDataset.csv in the same folder as this script.
# 3. Install dependencies:
#       pip install streamlit pandas scikit-learn plotly
# 4. Launch the app:
#       streamlit run ap2_frontend.py
# 5. Open http://localhost:8501 in your browser.
# 6. Use the sliders and dropdowns to explore driver clusters in 2D and 3D.
#   you may need to install == pip install matplotlib


