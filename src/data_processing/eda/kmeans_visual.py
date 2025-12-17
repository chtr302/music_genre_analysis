import pandas as pd, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/spotify_data_processed.csv"))
df = pd.read_csv(DATA_PATH)

features = [
    "streams","total_playlists","chart_appearances_count",
    "artist_avg_streams","danceability_%","energy_%","acousticness_%"
]

X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

# Elbow Method 
ks, inertias = range(2,6), []
for k in ks:
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_)

plt.plot(ks, inertias, marker="o")
plt.xlabel("So luong cluster (k)")
plt.ylabel("Inertia (do tap trung)")
plt.title("Elbow Method")
plt.show()

# Silhouette 
sils = []
for k in ks:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    sils.append(silhouette_score(X_scaled, labels))

plt.plot(ks, sils, marker="o")
plt.xlabel("So luong cluster (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score")
plt.show()

# PCA 2D 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df.loc[X.index, "PCA_Streams_Playlist"] = X_pca[:,0]
df.loc[X.index, "PCA_Audio_Features"] = X_pca[:,1]
df.loc[X.index, "cluster"] = labels

cluster_names = {
    0: "Hit & Pho bien", 1: "Pho bien vua", 2: "Niche / Acoustic"
}
df["cluster_name"] = df["cluster"].map(cluster_names)

sns.scatterplot(
    data=df,
    x="PCA_Streams_Playlist",
    y="PCA_Audio_Features",
    hue="cluster_name",
    palette="Set2",
    alpha=0.7
)

plt.title("KMeans Clustering (PCA 2D)")
plt.xlabel("Tong hop Streams + Playlist")
plt.ylabel("Tong hop Audio Features")
plt.legend(title="Loai cum")
plt.show()
