import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

#path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"../../../data/spotify_clustered.csv")

#load data
df = pd.read_csv(DATA_PATH)

#features
features = ["streams","in_spotify_playlists","in_spotify_charts",
            "danceability_%","energy_%","acousticness_%"]
features = [f for f in features if f in df.columns]

#standardize
X_scaled = StandardScaler().fit_transform(df[features])

#pca
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]

#scatter 
plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df,
    x="PCA1", y="PCA2",
    hue="cluster_name",
    palette="Set2",
    alpha=0.7
)
plt.title("Spotify Song Clusters PCA")
plt.xlabel("PCA1 – Do pho bien (Popularity)")
plt.ylabel("PCA2 – Dac trung audio (Audio features)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

#boxplot 
plt.figure(figsize=(8,5))
sns.boxplot(x="cluster_name", y="streams", data=df, palette="Set2")
plt.title("Phan bo Streams theo Cluster")
plt.xlabel("Cluster")
plt.ylabel("Streams")
plt.tight_layout()
plt.show()

#histogram 
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="streams", hue="cluster_name", palette="Set2", bins=50)
plt.title("Histogram Streams theo Cluster")
plt.xlabel("Streams")
plt.ylabel("So luong bai hat")
plt.tight_layout()
plt.show()
