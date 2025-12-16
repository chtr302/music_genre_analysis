import pandas as pd, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/spotify_data_processed.csv"))
df = pd.read_csv(DATA_PATH)

# Các feature thực sự cần cho clustering
features = [
    "streams","total_playlists","chart_appearances_count",
    "artist_avg_streams","danceability_%","energy_%","acousticness_%"
]

X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

# Thử nhiều k để đánh giá
print("Elbow + Silhouette")
for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"k={k}, inertia={km.inertia_:.2f}, silhouette={silhouette_score(X_scaled, labels):.3f}")

# Chọn k=3 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

# Trung bình feature theo cluster
print("\nTrung binh cac feature theo cluster")
print(df.groupby("cluster")[features].mean())

# Định nghĩa HIT: top 25% streams
df["is_hit"] = df["streams"] > df["streams"].quantile(0.75)
print("\nTy le HIT theo cluster")
print(df.groupby("cluster")["is_hit"].mean())
