import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def process_missing_value(data : pd.DataFrame) -> pd.DataFrame:
    """Hàm này xử lý các cột giá trị missing"""
    data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
    stream_median_value = data['streams'].median()
    data['streams'] = data['streams'].fillna(value=stream_median_value)
    data['log_streams'] = np.log1p(data['streams'])

    columns_to_process = ['in_deezer_playlists', 'in_shazam_charts']

    for column in columns_to_process:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].fillna(0).astype(int)

    data['key'] = data['key'].fillna(value='Unknown')

    return data

def process_categorical_column(data : pd.DataFrame, colums_categorical : list[str]) -> pd.DataFrame:
    """Hàm này sẽ encode các cột thuộc kiểu thể loại"""
    return pd.get_dummies(data=data, columns=colums_categorical, dtype=int)

def standardize_columns(data : pd.DataFrame, columns_standardize : list[str]) -> pd.DataFrame:
    """Đây là hàm để chuẩn hóa các giá trị đầu vào sử dụng Standard"""
    data_standardize = data.copy()
    scaler = StandardScaler()

    data_standardize[columns_standardize] = scaler.fit_transform(data_standardize[columns_standardize])

    return data_standardize

def create_new_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Dựa vào các cột để tạo ra đặc trưng mới"""

    # 1. artist_avg_streams
    all_artists = data.copy()
    all_artists['artist(s)_name'] = all_artists['artist(s)_name'].str.split(', ')
    exploded_artists = all_artists.explode('artist(s)_name')
    
    # Tính lượt stream trung bình cho mỗi nghệ sĩ
    artist_avg_streams_map = exploded_artists.groupby('artist(s)_name')['streams'].mean()

    # Hàm để tính giá trị trung bình cho mỗi bài hát
    def get_artist_avg_streams(artists):
        artist_names = [name.strip() for name in artists.split(',')]
        avg_streams = [artist_avg_streams_map.get(name, 0) for name in artist_names]
        return np.mean(avg_streams) if avg_streams else 0

    data['artist_avg_streams'] = data['artist(s)_name'].apply(get_artist_avg_streams)

    # 2. total_playlists
    data['total_playlists'] = data['in_spotify_playlists'] + data['in_apple_playlists'] + data['in_deezer_playlists']

    # 3. chart_appearances_count
    chart_columns = ['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']
    for col in chart_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0) # Điền 0 cho các giá trị không phải số

    data['chart_appearances_count'] = data[chart_columns].apply(lambda row: (row > 0).sum(), axis=1)
    
    return data

if __name__ == "__main__":
    data_raw = pd.read_csv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "spotify-2023.csv")),
        encoding='latin-1',
        thousands=','
    )
    data_cleaned = process_missing_value(data_raw)
    data_with_features = create_new_feature(data_cleaned.copy())
    data_encoded = process_categorical_column(data_with_features, ['key', 'mode'])
    data_encoded['release_date'] = pd.to_datetime(
        data_encoded['released_year'].astype(str) + '-' +
        data_encoded['released_month'].astype(str) + '-' +
        data_encoded['released_day'].astype(str)
    )
    data_unstandardized = data_encoded.drop(columns=['released_year', 'released_month', 'released_day'])
    columns_to_standardize = [
        'artist_count', 'in_spotify_playlists', 'in_spotify_charts', 'streams', 
        'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 
        'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 
        'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%', 
        'artist_avg_streams', 'total_playlists', 'chart_appearances_count'
    ]
    data_standardized = standardize_columns(data_unstandardized, columns_to_standardize)

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "spotify_data_processed.csv"))
    data_standardized.to_csv(csv_path, index=False)