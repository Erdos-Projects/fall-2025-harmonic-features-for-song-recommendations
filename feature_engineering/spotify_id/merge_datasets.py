"""
Merge chordonomicon with several Kaggle datasets based on Spotify song IDs.
"""

import os
import pandas as pd
from pathlib import Path


data_dir = Path("../../data")
# print(os.listdir(data_dir))


def merge_dataset_1(unique_ids_df):

    data_tracks = pd.read_csv(data_dir / "spotify-12m-songs" / "tracks_features.csv", low_memory=False)

    # print(len(data_tracks))
    # # 1204025

    # print(data_tracks.columns)
    # # ['id', 'name', 'album', 'album_id', 'artists', 'artist_ids',
    # #        'track_number', 'disc_number', 'explicit', 'danceability', 'energy',
    # #        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    # #        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
    # #        'time_signature', 'year', 'release_date']

    # duplicates = data_tracks[data_tracks.duplicated(subset=['id'], keep=False)]
    # print(f"Number of duplicates: {duplicates.shape[0]}")
    # # 0

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='id',
        how='left'
    )

    # not_merged = merged_df[merged_df['id'].isna()]
    # print(len(not_merged))
    # # 397119

    # print(merged_df['id'].notna().sum())
    # # 33204


def merge_dataset_2(unique_ids_df):

    # https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/code

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


def merge_dataset_3(unique_ids_df):

    data_tracks = pd.read_csv(data_dir / "-spotify-tracks-dataset" / "dataset.csv", low_memory=False)

    # print(len(data_tracks))
    # # 114000

    # print(data_tracks.columns)
    # # ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name',
    # #        'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    # #        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    # #        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
    # #        'track_genre']

    # duplicates = data_tracks[data_tracks.duplicated(subset=['track_id'], keep=False)]
    # print(f"Number of duplicates: {duplicates.shape[0]}")
    # # 40900

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='track_id',
        how='left'
    )

    # not_merged = merged_df[merged_df['track_id'].isna()]
    # print(len(not_merged))
    # # 421244

    # print(merged_df['track_id'].notna().sum())
    # # 12426


def merge_dataset_4(unique_ids_df):

    data_tracks = pd.read_csv(data_dir / "ultimate-spotify-tracks-db" / "SpotifyFeatures.csv", low_memory=False)

    print(len(data_tracks))
    # 232725

    print(data_tracks.columns)
    # ['genre', 'artist_name', 'track_name', 'track_id', 'popularity',
    #        'acousticness', 'danceability', 'duration_ms', 'energy',
    #        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
    #        'speechiness', 'tempo', 'time_signature', 'valence']

    duplicates = data_tracks[data_tracks.duplicated(subset=['track_id'], keep=False)]
    print(f"Number of duplicates: {duplicates.shape[0]}")
    # 91075

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='track_id',
        how='left'
    )

    not_merged = merged_df[merged_df['track_id'].isna()]
    print(len(not_merged))
    # 404826

    print(merged_df['track_id'].notna().sum())
    # 41214


def merge_dataset_5(unique_ids_df):

    data_tracks = pd.read_csv(data_dir / "spotify-tracks-genre-dataset" / "train.csv", low_memory=False)

    print(len(data_tracks))
    # 114000

    print(data_tracks.columns)
    # ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name',
    #        'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    #        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    #        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
    #        'track_genre']

    duplicates = data_tracks[data_tracks.duplicated(subset=['track_id'], keep=False)]
    print(f"Number of duplicates: {duplicates.shape[0]}")
    # 40900

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='track_id',
        how='left'
    )

    not_merged = merged_df[merged_df['track_id'].isna()]
    print(len(not_merged))
    # 421244

    print(merged_df['track_id'].notna().sum())
    # 12426


def merge_dataset_6(unique_ids_df):

    data_tracks_1 = pd.read_csv(data_dir / "spotify-music-dataset" / "high_popularity_spotify_data.csv", low_memory=False)
    data_tracks_2 = pd.read_csv(data_dir / "spotify-music-dataset" / "low_popularity_spotify_data.csv", low_memory=False)

    # Ensure both DataFrames have the same columns (order-agnostic)
    cols_1 = set(data_tracks_1.columns)
    cols_2 = set(data_tracks_2.columns)

    if cols_1 == cols_2:
        # Align column order to `data_tracks_1` and concatenate vertically
        data_tracks_2 = data_tracks_2.reindex(columns=list(data_tracks_1.columns))
        data_tracks = pd.concat([data_tracks_1, data_tracks_2], ignore_index=True)
        print(f"Concatenated shape: {data_tracks.shape}")
    else:
        missing_in_2 = sorted(cols_1 - cols_2)
        missing_in_1 = sorted(cols_2 - cols_1)
        raise ValueError(
            f"Column mismatch.\n"
            f"Missing in second: {missing_in_2}\n"
            f"Missing in first: {missing_in_1}"
        )

    print(len(data_tracks))
    # 4831

    print(data_tracks.columns)
    # ['energy', 'tempo', 'danceability', 'playlist_genre', 'loudness',
    #        'liveness', 'valence', 'track_artist', 'time_signature', 'speechiness',
    #        'track_popularity', 'track_href', 'uri', 'track_album_name',
    #        'playlist_name', 'analysis_url', 'track_id', 'track_name',
    #        'track_album_release_date', 'instrumentalness', 'track_album_id',
    #        'mode', 'key', 'duration_ms', 'acousticness', 'id', 'playlist_subgenre',
    #        'type', 'playlist_id']

    duplicates = data_tracks[data_tracks.duplicated(subset=['track_id'], keep=False)]
    print(f"Number of duplicates: {duplicates.shape[0]}")
    # 631

    merged_df = pd.merge(
        unique_ids_df,
        data_tracks,
        left_on='spotify_song_id',
        right_on='track_id',
        how='left'
    )

    not_merged = merged_df[merged_df['track_id'].isna()]
    print(len(not_merged))
    # 429669

    print(merged_df['track_id'].notna().sum())
    # 714


def main():

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

    # merge_dataset_1(unique_ids_df)

    # merge_dataset_2(unique_ids_df)

    # merge_dataset_3(unique_ids_df)

    # merge_dataset_4(unique_ids_df)

    # merge_dataset_5(unique_ids_df)

    merge_dataset_6(unique_ids_df)


if __name__ == "__main__":
    results = main()
