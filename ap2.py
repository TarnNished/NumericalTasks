import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("F1DriversDataset.csv")
print(data.columns)

features = ["Race_Wins", "Podiums", "Points", "Pole_Positions", "Fastest_Laps"]
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

print(data[["Driver", "Cluster"]])
print("\nCluster Centers:\n", pd.DataFrame(centers, columns=features))

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=data["Cluster"], cmap="viridis")
plt.xlabel("Wins (scaled)")
plt.ylabel("Points (scaled)")
plt.title("F1 Drivers Clustering — KMeans")
plt.grid(True)
plt.show()
