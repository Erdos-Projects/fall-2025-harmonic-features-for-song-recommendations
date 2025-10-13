"""
Merge new Kaggle dataset with existing dataset.
https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/code
"""

import os
import pandas as pd
from pathlib import Path



def main():

    data_dir = Path("../../data")
    # print(os.listdir(data_dir))

    data_chordonomicon = pd.read_csv(data_dir / "chordonomicon_v2.csv", low_memory=False)
    # print(len(data_chordonomicon))
    # # 679807

    # print(data_chordonomicon.columns)
    # # ['id', 'chords', 'release_date', 'genres', 'decade', 'rock_genre',
    # #        'artist_id', 'main_genre', 'spotify_song_id', 'spotify_artist_id']

    data_chordonomicon = data_chordonomicon.dropna(subset=['spotify_song_id'])
    # print(len(data_chordonomicon))
    # # 440284

    # duplicates = data_chordonomicon[data_chordonomicon.duplicated(subset=['spotify_song_id'], keep=False)]
    # print(f"Number of duplicates: {duplicates.shape[0]}")
    # # Number of duplicates: 18103
    # duplicates.to_csv(data_dir / "chordonomicon_v2_duplicates.csv", index=False)

    # # save the unique spotify song ids to csv
    unique_ids = data_chordonomicon['spotify_song_id'].drop_duplicates()
    unique_ids_df = pd.DataFrame({'spotify_song_id': unique_ids})
    # print(len(unique_ids_df))
    # # 430323
    # unique_ids_df.to_csv(data_dir / "chordonomicon_v2_unique_spotify_song_ids.csv", index=False)

    data_tracks = pd.read_csv(data_dir / "spotify-dataset-19212020-600k-tracks" / "tracks.csv", low_memory=False)
    # print(len(data_tracks))
    # # 586672

    # print(data_tracks.columns)
    # # ['id', 'name', 'popularity', 'duration_ms', 'explicit', 'artists',
    # #        'id_artists', 'release_date', 'danceability', 'energy', 'key',
    # #        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    # #        'liveness', 'valence', 'tempo', 'time_signature']

    # duplicates = data_tracks[data_tracks.duplicated(subset=['id'], keep=False)]
    # print(f"Number of duplicates: {duplicates.shape[0]}")
    # # 0

    # merged_df = pd.merge(
    #     data_chordonomicon,
    #     data_tracks,
    #     left_on='spotify_song_id',
    #     right_on='id',
    #     how='left'
    # )

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='id',
        how='left'
    )

    # not_merged = merged_df[merged_df['id'].isna()]
    # print(len(not_merged))
    # # 374359

    # print(merged_df['id'].notna().sum())
    # # 55964




if __name__ == "__main__":
    results = main()
