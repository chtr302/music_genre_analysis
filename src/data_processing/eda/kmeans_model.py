import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/spotify-2023.csv")
OUT_PATH = os.path.join(BASE_DIR, "../../../data/spotify_clustered.csv")
ELBOW_PLOT = os.path.join(BASE_DIR, "../../../data/elbow_plot.png")
SILHOUETTE_PLOT = os.path.join(BASE_DIR, "../../../data/silhouette_plot.png")

# read csv 
def read_csv_safe(path):
    for enc in ["utf-8", "latin1", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("Cannot read CSV")

df = read_csv_safe(DATA_PATH)

# clean numeric 
num_cols = [
    "streams",
    "in_spotify_playlists",
    "in_spotify_charts",
    "danceability_%",
    "energy_%", 
    "acousticness_%", 
    "instrumentalness_%", 
    "valence_%"
]

for col in num_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )

df = df.dropna(subset=["streams"])

# audio features
audio_features = [
    "danceability_%", "energy_%", "acousticness_%", "instrumentalness_%", "valence_%"
]

X = df[audio_features].dropna()

# standardize 
X_scaled = StandardScaler().fit_transform(X)

# Elbow & Silhouette để chọn k 
ks = range(2, 6)
inertia_list = []
silhouette_list = []

print("\n===== ELBOW & SILHOUETTE SCORES =====")
print("k\tInertia\t\tSilhouette")

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    inertia_list.append(inertia)
    silhouette_list.append(silhouette)
    print(f"{k}\t{inertia:.2f}\t\t{silhouette:.4f}")

# Biểu đồ Elbow
plt.figure(figsize=(6,4))
plt.plot(ks, inertia_list, marker='o', color='blue')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for KMeans')
plt.xticks(ks)
plt.grid(True)
plt.tight_layout()
plt.savefig(ELBOW_PLOT)
plt.show()
print(f"Elbow plot saved: {ELBOW_PLOT}")

# Silhouette
plt.figure(figsize=(6,4))
plt.plot(ks, silhouette_list, marker='s', color='red')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.xticks(ks)
plt.grid(True)
plt.tight_layout()
plt.savefig(SILHOUETTE_PLOT)
plt.show()
print(f"Silhouette plot saved: {SILHOUETTE_PLOT}")

# Chọn k = 3
k_final = 3
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

# hit label 
df["is_hit"] = df["streams"] >= df["streams"].quantile(0.75)

# rename clusters 
cluster_mean = df.groupby("cluster")[audio_features].mean()

cluster_map = {}
for c in cluster_mean.index:
    row = cluster_mean.loc[c]
    if row["acousticness_%"] > 50:
        cluster_map[c] = "Soft / Acoustic"
    elif row["energy_%"] > 60 and row["valence_%"] > 50:
        cluster_map[c] = "Energetic / Feel-Good"
    else:
        cluster_map[c] = "Popular / Mainstream"

df["cluster_name"] = df["cluster"].map(cluster_map)

print("\n===== Trung bình các đặc trưng theo cluster =====")
print(df.groupby("cluster_name")[audio_features].mean().round(2))

print("\n===== Tỉ lệ hit theo cluster =====")
print(df.groupby("cluster_name")["is_hit"].mean().round(3))

print("\n===== Playlist & Chart trung bình theo cluster =====")
print(
    df.groupby("cluster_name")[["in_spotify_playlists", "in_spotify_charts"]]
    .mean()
    .round(2)
)

# save
df[
    audio_features
    + ["streams", "in_spotify_playlists", "in_spotify_charts"]
    + ["cluster", "cluster_name"]
].dropna().to_csv(OUT_PATH, index=False)
print(f"\nClustered data saved: {OUT_PATH}")
