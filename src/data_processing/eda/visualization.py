import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# config
sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/spotify-2023.csv")

# read csv safely
def read_csv_safe(path):
    for enc in ["utf-8", "latin1", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("Cannot read CSV")

df = read_csv_safe(DATA_PATH)

# audio features
audio_features = [
    "bpm",
    "danceability_%",
    "energy_%",
    "valence_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%"
]

# clean numeric
for col in audio_features:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )

# drop missing
df_audio = df[audio_features].dropna()

# correlation matrix
corr_audio = df_audio.corr()

# plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_audio,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Correlation Between Audio Features", fontsize=14)
plt.tight_layout()
plt.show()
