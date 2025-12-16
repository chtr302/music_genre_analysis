import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/spotify_data_processed.csv"))
df = pd.read_csv(DATA_PATH)

features = [
    "danceability_%", "energy_%", "acousticness_%",
    "valence_%", "bpm", "artist_avg_streams", "total_playlists"
]

df_feat = df[features].fillna(0)

# scale
scaler = StandardScaler()
X = scaler.fit_transform(df_feat)

# kmeans (tao the loai ngam)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# hit
hit_threshold = df["streams"].quantile(0.75)
df["is_hit"] = df["streams"] >= hit_threshold

# chon hit
def get_representative_hits():
    reps = []
    for c in sorted(df["cluster"].unique()):
        hit_songs = df[(df["cluster"] == c) & (df["is_hit"])]
        if len(hit_songs) == 0:
            continue
        top_song = hit_songs.sort_values("streams", ascending=False).iloc[0]
        reps.append(top_song)
    return pd.DataFrame(reps)

# goi y bai hat tuong tu
similarity_matrix = cosine_similarity(X, X)

def recommend(song_index, top_k=5):
    scores = list(enumerate(similarity_matrix[song_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    idx = [i for i, _ in scores]
    return df.loc[idx, ["track_name", "artist(s)_name", "streams"]]

# main
if __name__ == "__main__":
    reps = get_representative_hits().reset_index()

    print("5 bai HIT dai dien cho 5 the loai:")
    for i, row in reps.iterrows():
        print(f"{i}. {row['track_name']} - {row['artist(s)_name']}")

    choice = int(input("\nChon 1 bai (0-4): "))
    song_idx = reps.loc[choice, "index"]

    print("\nBai ban chon:")
    # in ra thông tin cơ bản + các đặc trưng
    display_cols = ["track_name", "artist(s)_name", "streams"] + features
    print(df.loc[song_idx, display_cols])

    print("\nGoi y bai tuong tu:")
    print(recommend(song_idx, top_k=5))
