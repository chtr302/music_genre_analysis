import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/spotify_data_processed.csv")
df = pd.read_csv(DATA_PATH)

sns.set(style="whitegrid")

# HISTOGRAM
if "streams" in df.columns:
    plt.figure(figsize=(7,4))
    sns.histplot(df["streams"], kde=True)
    plt.title("Distribution of Streams")
    plt.xlabel("Streams")
    plt.ylabel("Frequency")
    plt.show()

#SCATTER
for col in ["energy_%", "danceability_%", "acousticness_%"]:
    if col in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df["streams"], alpha=0.5)
        plt.title(f"Streams vs {col}")
        plt.xlabel(col)
        plt.ylabel("Streams")
        plt.show()

# BOXPLOT
if "cluster" in df.columns:
    plt.figure(figsize=(7,4))
    sns.boxplot(x=df["cluster"], y=df["streams"])
    plt.title("Streams Distribution by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Streams")
    plt.show()

# HEATMAP 
num_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,7))
sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
