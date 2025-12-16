import pandas as pd, numpy as np, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/spotify_data_processed.csv"))

df = pd.read_csv(DATA_PATH)

features = ["danceability_%","energy_%","acousticness_%","valence_%","bpm","artist_avg_streams","total_playlists"]
df_feat = df[features].fillna(0)

scaler = StandardScaler()
X = scaler.fit_transform(df_feat)

similarity_matrix = cosine_similarity(X, X)

def recommend(song_index, top_k=10):
    scores = list(enumerate(similarity_matrix[song_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    rec_idx = [i for i,_ in scores]
    return df.loc[rec_idx, ["track_name","artist(s)_name","streams"]]

if __name__ == "__main__":
    print("Bai goc:")
    print(df.loc[0, ["track_name","artist(s)_name"]])
    print("\nDanh sach goi y:")
    print(recommend(0, top_k=5))
