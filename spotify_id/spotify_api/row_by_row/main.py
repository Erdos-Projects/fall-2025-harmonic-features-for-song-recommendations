"""
Spotify Data Enrichment Pipeline - Sequential Processing

Processes song data one at a time using the Spotify API.

Usage:
    python main.py

Configuration can be modified in the Configuration section below.
"""

from spotify_pipeline import spotify_enrichment_pipeline


def main():
    """Main entry point for Spotify data enrichment"""

    # ========================================
    # Configuration
    # ========================================

    # File paths
    input_file = "chordonomicon_part_2.csv"
    output_file = "chordonomicon_part_2_enriched.csv"

    # Processing options
    get_track_info = True       # Track name, album, release date, popularity
    get_artist_info = True      # Artist name, genres, followers
    get_audio_features = False  # Audio features (key, mode, tempo, danceability, etc.)

    # Processing control
    num_rows = 10               # None = all rows, or set number for testing

    # ========================================
    # Run Pipeline
    # ========================================

    results_df = spotify_enrichment_pipeline(
        input_file=input_file,
        output_file=output_file,
        get_track_info=get_track_info,
        get_artist_info=get_artist_info,
        get_audio_features=get_audio_features,
        num_rows=num_rows
    )

    # ========================================
    # Display Results Summary
    # ========================================

    if results_df is not None:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        # Basic info
        print(f"Shape: {results_df.shape}")
        print(f"Columns: {list(results_df.columns)}")

        # Check for failures
        if 'success' in results_df.columns:
            failed = results_df[results_df['success'] == False]
            print(f"\nFailed rows: {len(failed)}")

        # Display first few rows
        print("\nFirst few rows:")
        print(results_df.head().to_string())

        # Summary statistics if track info was fetched
        if get_track_info and 'popularity' in results_df.columns:
            print("\n" + "=" * 60)
            print("Popularity Statistics")
            print("=" * 60)
            print(results_df['popularity'].describe())

        # Genre analysis if artist info was fetched
        if get_artist_info and 'genres' in results_df.columns:
            print("\n" + "=" * 60)
            print("Most Common Genres")
            print("=" * 60)
            all_genres = []
            for genres_str in results_df['genres'].dropna():
                if genres_str:
                    all_genres.extend(genres_str.split(', '))

            if all_genres:
                from collections import Counter
                genre_counts = Counter(all_genres)
                for genre, count in genre_counts.most_common(10):
                    print(f"  {genre}: {count}")
            else:
                print("  No genre data available")

    # ========================================
    # Policy Notice
    # ========================================

    print("\n" + "=" * 60)
    print("IMPORTANT: Spotify Policy Notice")
    print("=" * 60)
    print("Spotify content may not be used to train machine learning")
    print("or AI models. You cannot use the Spotify Platform or any")
    print("Spotify Content to train ML/AI models or ingest Spotify")
    print("Content into ML/AI models.")
    print("=" * 60 + "\n")

    return results_df


if __name__ == "__main__":
    results = main()
