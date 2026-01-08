import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

#  path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/spotify_clustered.csv")

#  load data
df = pd.read_csv(DATA_PATH)

# audio features 
audio_features = [
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "valence_%"
]

#  standardize 
X_scaled = StandardScaler().fit_transform(df[audio_features])

# pca 
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# plot 
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="cluster_name",
    palette="Set2",
    s=20,
    alpha=0.7
)

plt.title("KMeans Clustering of Spotify Songs (Audio Features)", fontsize=13)
plt.xlabel("PCA 1 – Energy & Rhythm")
plt.ylabel("PCA 2 – Mood & Acoustic")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
