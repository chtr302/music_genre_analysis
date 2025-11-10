**track_name**: Name of the song
**artist(s)_name**: Name of the artist(s) of the song
**artist_count**: Number of artists contributing to the song
**released_year**: Year when the song was released
**released_monthd**: Month when the song was release
**released_day**: Day of the month when the song was released
**in_spotify_playlist**: Number of Spotify playlists the song is included in
**in_spotify_charts**: Presence and rank of the song on Spotify charts
**streams**: Total number of streams on Spotify
**in_apple_playlists**: Number of Apple Music playlists the song is included in
**in_apple_charts**: Presence and rank of the song on Apple Music charts
**in_deezer_playlists**: Number of Deezer playlists the song is included in
**in_deezer_charts**: Presence and rank of the song on Deezer charts
**in_shazam_chart**s: Presence and rank of the song on Shazam charts
**bpm**: Beats per minute, a measure of song tempo
**key**: Key of the song
**mode**: Mode of the song (major or minor)**
**danceability_%**: Percentage indicating how suitable the song is for dancing
**valence_%**: Positivity of the song's musical content
**energy_%**: Perceived energy level of the song
**acousticness_%**: Amount of acoustic sound in the song
**instrumentalness_%**: Amount of instrumental content in the song
**liveness_%**: Presence of live performance elements
**speechiness_%**: Amount of spoken words in the song
---

## Generated Data Files

### `spotify_data_processed.xlsx`
- **Purpose:** This file contains the processed and feature-engineered dataset **before standardization**. It includes newly created features for deeper analysis.
- **Sheets:**
    - `data`: The main dataset.
    - `column_descriptions`: Explanations for the newly created columns.
- **Use Case:** Ideal for exploratory data analysis (EDA), understanding feature distributions, and verifying the feature creation logic.

### `spotify_data_processed.csv`
- **Purpose:** This file contains the final, model-ready dataset. It includes all created features, and the numerical columns have been **standardized** (scaled to have a mean of ~0 and a standard deviation of ~1).
- **Use Case:** Intended as the direct input for training machine learning models.