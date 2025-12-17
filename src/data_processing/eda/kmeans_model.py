import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"../../../data/spotify-2023.csv")
OUT_PATH = os.path.join(BASE_DIR,"../../../data/spotify_clustered.csv")

def read_csv_safe(path):
    for enc in ["utf-8","latin1","utf-16"]:
        try:
            return pd.read_csv(path,encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("khong doc duoc csv")

df = read_csv_safe(DATA_PATH)

#clean streams
df["streams"] = df["streams"].astype(str).str.replace(",","",regex=False)
df["streams"] = pd.to_numeric(df["streams"],errors="coerce")
df = df.dropna(subset=["streams"])

#features
features = ["streams","in_spotify_playlists","in_spotify_charts",
            "danceability_%","energy_%","acousticness_%"]
features = [f for f in features if f in df.columns]
X = df[features].dropna()

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#elbow+silhouette
print("Elbow+Silhouette:")
for k in range(2,6):
    km = KMeans(n_clusters=k,random_state=42,n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"k={k} | inertia={km.inertia_:.0f} | silhouette={silhouette_score(X_scaled,labels):.3f}")

#train final model
kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
df.loc[X.index,"cluster"] = kmeans.fit_predict(X_scaled)

#is_hit
df["is_hit"] = df["streams"]>=df["streams"].quantile(0.75)

#cluster_name
hit_rate = df.groupby("cluster")["is_hit"].mean().sort_values()
cluster_map = {
    hit_rate.index[0]:"Niche/Acoustic",
    hit_rate.index[1]:"Popular",
    hit_rate.index[2]:"Hit&Mainstream"
}
df["cluster_name"] = df["cluster"].map(cluster_map)

#summary
print("\ntrungbinh dac trung theo cluster")
print(df.groupby("cluster_name")[features].mean().round(2))
print("\nty le hit theo cluster")
print(df.groupby("cluster_name")["is_hit"].mean().round(3))

#save
df[features+["cluster","cluster_name"]].dropna().to_csv(OUT_PATH,index=False)
print(f"\nda luu file: {OUT_PATH}")
