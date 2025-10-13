"""
EDA: Extract per-dataset Spotify track IDs, combine them, dedupe, and merge with chordonomicon unique IDs.
"""

import pandas as pd
from pathlib import Path

data_dir = Path("../../data")


def get_ids_dataset_1():
    # spotify-12m-songs
    return (
        pd.read_csv(data_dir / "spotify-12m-songs" / "tracks_features.csv", low_memory=False)["id"]
        .dropna().drop_duplicates().rename("spotify_song_id")
    )


def get_ids_dataset_2():
    # spotify-dataset-19212020-600k-tracks
    return (
        pd.read_csv(data_dir / "spotify-dataset-19212020-600k-tracks" / "tracks.csv", low_memory=False)["id"]
        .dropna().drop_duplicates().rename("spotify_song_id")
    )


def get_ids_dataset_3():
    # -spotify-tracks-dataset
    return (
        pd.read_csv(data_dir / "-spotify-tracks-dataset" / "dataset.csv", low_memory=False)["track_id"]
        .dropna().drop_duplicates().rename("spotify_song_id")
    )


def get_ids_dataset_4():
    # ultimate-spotify-tracks-db
    return (
        pd.read_csv(data_dir / "ultimate-spotify-tracks-db" / "SpotifyFeatures.csv", low_memory=False)["track_id"]
        .dropna().drop_duplicates().rename("spotify_song_id")
    )


def get_ids_dataset_5():
    # spotify-tracks-genre-dataset
    return (
        pd.read_csv(data_dir / "spotify-tracks-genre-dataset" / "train.csv", low_memory=False)["track_id"]
        .dropna().drop_duplicates().rename("spotify_song_id")
    )


def get_ids_dataset_6():
    # spotify-music-dataset (high + low popularity)
    d1 = pd.read_csv(data_dir / "spotify-music-dataset" / "high_popularity_spotify_data.csv", low_memory=False)
    d2 = pd.read_csv(data_dir / "spotify-music-dataset" / "low_popularity_spotify_data.csv", low_memory=False)
    all_d = pd.concat([d1, d2], ignore_index=True)
    return all_d["track_id"].dropna().drop_duplicates().rename("spotify_song_id")


def main():
    # chordonomicon unique Spotify IDs
    chord = pd.read_csv(data_dir / "chordonomicon_v2.csv", low_memory=False)
    chord = chord.dropna(subset=["spotify_song_id"])
    unique_ids_df = pd.DataFrame({"spotify_song_id": chord["spotify_song_id"].drop_duplicates()})

    # per-dataset ID lists
    ids_1 = get_ids_dataset_1()
    ids_2 = get_ids_dataset_2()
    ids_3 = get_ids_dataset_3()
    ids_4 = get_ids_dataset_4()
    ids_5 = get_ids_dataset_5()
    ids_6 = get_ids_dataset_6()

    # save per-dataset lists
    ids_1.to_frame().to_csv(data_dir / "dataset_1_track_ids.csv", index=False)
    ids_2.to_frame().to_csv(data_dir / "dataset_2_track_ids.csv", index=False)
    ids_3.to_frame().to_csv(data_dir / "dataset_3_track_ids.csv", index=False)
    ids_4.to_frame().to_csv(data_dir / "dataset_4_track_ids.csv", index=False)
    ids_5.to_frame().to_csv(data_dir / "dataset_5_track_ids.csv", index=False)
    ids_6.to_frame().to_csv(data_dir / "dataset_6_track_ids.csv", index=False)

    # combine, dedupe, and merge with unique_ids_df
    combined_ids = pd.concat([ids_1, ids_2, ids_3, ids_4, ids_5, ids_6], ignore_index=True).drop_duplicates()
    combined_ids_df = pd.DataFrame({"spotify_song_id": combined_ids})
    combined_ids_df.to_csv(data_dir / "combined_track_ids.csv", index=False)

    merged = unique_ids_df.merge(
        combined_ids_df.assign(in_any_dataset=1),
        on="spotify_song_id",
        how="left",
    )
    merged.to_csv(data_dir / "unique_ids_merged_with_combined.csv", index=False)

    # simple coverage stats
    total_unique = len(unique_ids_df)
    represented = merged["in_any_dataset"].notna().sum()

    print(f"Chordonomicon unique ids: {total_unique}")
    print(f"Combined dataset unique ids: {len(combined_ids_df)}")
    print(f"Represented in combined: {represented} ({represented/total_unique:.2%})")

    # Chordonomicon unique ids: 430323
    # Combined dataset unique ids: 1963081
    # Represented in combined: 99970 (23.23%)


if __name__ == "__main__":
    main()
