import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"../../../data/spotify-2023.csv")

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

#histogram
plt.figure(figsize=(8,5))
sns.histplot(df["streams"],bins=50,kde=True)
plt.title("Phanphoi do pho bien bai hat (Streams goc)")
plt.xlabel("Streams")
plt.ylabel("So luong bai hat")
plt.tight_layout()
plt.show()

#scatter audio features
audio_features = ["danceability_%","energy_%","acousticness_%"]
fig,axes = plt.subplots(1,3,figsize=(15,4))
for i,f in enumerate(audio_features):
    if f in df.columns:
        sns.scatterplot(x=df[f],y=df["streams"],alpha=0.4,ax=axes[i])
        axes[i].set_title(f"{f} vs Streams")
        axes[i].set_xlabel(f)
        axes[i].set_ylabel("Streams")
plt.tight_layout()
plt.show()

#heatmap
corr_features = ["streams","in_spotify_playlists","in_spotify_charts",
                 "danceability_%","energy_%","acousticness_%"]
corr_features = [f for f in corr_features if f in df.columns]

plt.figure(figsize=(8,6))
sns.heatmap(df[corr_features].corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0)
plt.title("Correlation giua Streams va Audio Features")
plt.tight_layout()
plt.show()
