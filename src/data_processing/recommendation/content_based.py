import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/spotify-2023.csv"))

# doc csv, tu thu nhieu encoding
def read_csv_safe(path):
    for enc in ["utf-8", "latin1", "utf-16"]:
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: pass
    raise ValueError("Khong doc duoc file CSV")

df = read_csv_safe(DATA_PATH)

# chuan hoa cot streams ve so
df["streams"] = pd.to_numeric(
    df["streams"].astype(str).str.replace(",", ""),
    errors="coerce"
).fillna(0)

# cac feature dung cho content-based
features_all = [
    "danceability_%", "energy_%", "acousticness_%",
    "valence_%", "bpm", "artist_avg_streams", "total_playlists"
]
features = [f for f in features_all if f in df.columns]
print("Features used:", features)

# scale du lieu
X = StandardScaler().fit_transform(df[features].fillna(0))

# phan cum kmeans (the loai ngam)
df["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X)

# xac dinh bai hit (top 25% streams)
hit_th = df["streams"].quantile(0.75)
df["is_hit"] = df["streams"] >= hit_th

# chon bai hit dai dien moi cluster
def get_representative_hits():
    reps = []
    for c in sorted(df["cluster"].unique()):
        hits = df[(df["cluster"] == c) & (df["is_hit"])]
        if len(hits): reps.append(hits.sort_values("streams", ascending=False).iloc[0])
    return pd.DataFrame(reps)

# tinh do tuong tu cosine
sim_matrix = cosine_similarity(X)

# goi y bai hat tuong tu
def recommend(idx, k=5):
    sims = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:k+1]
    return df.loc[[i for i, _ in sims], ["track_name", "artist(s)_name", "streams"]]

# main
if __name__ == "__main__":
    reps = get_representative_hits().reset_index()

    print("5 bai HIT dai dien cho 5 the loai:")
    for i, r in reps.iterrows():
        print(f"{i}. {r['track_name']} - {r['artist(s)_name']}")

    c = int(input("\nChon bai (0-4): "))
    idx = reps.loc[c, "index"]

    print("\nBai ban chon:")
    print(df.loc[idx, ["track_name", "artist(s)_name", "streams"] + features])

    print("\nGoi y bai tuong tu:")
    print(recommend(idx))
