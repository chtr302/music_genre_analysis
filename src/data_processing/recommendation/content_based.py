import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#loaddata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/spotify-2023.csv")

def read_csv_safe(path):
    for enc in ["utf-8", "latin1", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("Cannot read CSV")

df = read_csv_safe(DATA_PATH)

#audiofeatures
audio_features = [
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "valence_%"
]

#preprocess
cols = ["streams"] + audio_features
for col in cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

df = df.dropna(subset=audio_features + ["streams"])

#standardize
X = df[audio_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

#hitlabel
df["is_hit"] = df["streams"] >= df["streams"].quantile(0.75)

#representativehits
def get_representative_hits():
    reps = []
    for c in sorted(df["cluster"].unique()):
        hit = (
            df[(df["cluster"] == c) & (df["is_hit"])]
            .sort_values("streams", ascending=False)
            .iloc[0]
        )
        reps.append(hit)
    return pd.DataFrame(reps)

#recommend
def recommend(song_idx, top_k=5):
    song_cluster = df.loc[song_idx, "cluster"]

    idx_cluster = df[df["cluster"] == song_cluster].index
    X_cluster = X_scaled[idx_cluster]
    target_vec = X_scaled[song_idx].reshape(1, -1)

    sims = cosine_similarity(target_vec, X_cluster)[0]

    sim_df = pd.DataFrame({
        "idx": idx_cluster,
        "sim": sims
    }).sort_values("sim", ascending=False)

    sim_df = sim_df[sim_df["idx"] != song_idx].head(top_k)

    return df.loc[
        sim_df["idx"],
        ["track_name", "artist(s)_name", "streams"] + audio_features
    ]

#demo
if __name__ == "__main__":
    reps = get_representative_hits().reset_index()

    print("3 bai HIT dai dien cho 3 phong cach:")
    for i, r in reps.iterrows():
        print(f"{i}. {r['track_name']} - {r['artist(s)_name']}")

    c = int(input("Chon bai (0-2): "))
    idx = reps.loc[c, "index"]

    print("\nBai ban chon:")
    print(df.loc[idx, ["track_name", "artist(s)_name"] + audio_features])

    print("\nGoi y bai tuong tu:")
    print(recommend(idx))
