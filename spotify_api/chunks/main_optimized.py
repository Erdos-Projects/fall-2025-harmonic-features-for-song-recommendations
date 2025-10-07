"""
Spotify Data Enrichment Pipeline - Optimized

Processes song data in batches with checkpoints for efficiency and reliability.

Features:
- Batch API calls (50 tracks, 100 audio features at once)
- Artist caching to avoid duplicate lookups
- Checkpoint saves every 500 songs (configurable)
- Resume capability if interrupted

Usage:
    python main_optimized.py

Configuration can be modified in the Configuration section below.
"""

from spotify_pipeline_optimized import spotify_enrichment_pipeline_optimized


def main():
    """Main entry point for Spotify data enrichment"""

    # ========================================
    # Configuration
    # ========================================

    # File paths
    input_file = "chordonomicon_part_2.csv"
    output_dir = "output_spotify"
    output_dir_final = "output_spotify_final"
    base_output_name = "chordonomicon_part_2_enriched"

    # Processing options
    get_track_info = True       # Track name, album, release date, popularity
    get_artist_info = True      # Artist name, genres, followers
    get_audio_features = False  # Audio features (key, mode, tempo, danceability, etc.)

    # Processing control
    num_rows = None             # None = all rows, or set number for testing
    chunk_size = 500            # Songs per checkpoint
    resume = True               # Resume from last checkpoint if available

    # ========================================
    # Run Pipeline
    # ========================================

    results_df = spotify_enrichment_pipeline_optimized(
        input_file=input_file,
        output_dir=output_dir,
        output_dir_final=output_dir_final,
        base_output_name=base_output_name,
        get_track_info=get_track_info,
        get_artist_info=get_artist_info,
        get_audio_features=get_audio_features,
        num_rows=num_rows,
        chunk_size=chunk_size,
        resume=resume
    )

    # ========================================
    # Display Results Summary (if successful)
    # ========================================

    if results_df is not None:
        print("\n" + "=" * 80)
        print("Detailed Results Summary")
        print("=" * 80)

        # Basic info
        print(f"\nShape: {results_df.shape}")
        print(f"Columns ({len(results_df.columns)}): {list(results_df.columns)}")

        # Check for failures
        if 'success' in results_df.columns:
            failed = results_df[results_df['success'] == False]
            success_rate = (1 - len(failed) / len(results_df)) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
            print(f"Failed rows: {len(failed):,}")

        # Summary statistics if track info was fetched
        if get_track_info and 'popularity' in results_df.columns:
            print("\n" + "=" * 80)
            print("Popularity Statistics")
            print("=" * 80)
            print(results_df['popularity'].describe())

        # Genre analysis if artist info was fetched
        if get_artist_info and 'genres' in results_df.columns:
            print("\n" + "=" * 80)
            print("Most Common Genres (Top 15)")
            print("=" * 80)
            all_genres = []
            for genres_str in results_df['genres'].dropna():
                if genres_str:
                    all_genres.extend(genres_str.split(', '))

            if all_genres:
                from collections import Counter
                genre_counts = Counter(all_genres)
                for genre, count in genre_counts.most_common(15):
                    print(f"  {genre}: {count:,}")
            else:
                print("  No genre data available")

        # Audio features summary
        if get_audio_features and 'tempo' in results_df.columns:
            print("\n" + "=" * 80)
            print("Audio Features Summary")
            print("=" * 80)
            audio_cols = ['tempo', 'danceability', 'energy', 'valence', 'acousticness']
            available_cols = [col for col in audio_cols if col in results_df.columns]
            if available_cols:
                print(results_df[available_cols].describe())

        # Display first few rows
        print("\n" + "=" * 80)
        print("Sample Data (first 5 rows)")
        print("=" * 80)
        display_cols = ['track_name', 'artists', 'popularity', 'genres']
        display_cols = [col for col in display_cols if col in results_df.columns]
        if display_cols:
            print(results_df[display_cols].head().to_string())

    # ========================================
    # Policy Notice
    # ========================================

    print("\n" + "=" * 80)
    print("IMPORTANT: Spotify Policy Notice")
    print("=" * 80)
    print("Spotify content may not be used to train machine learning")
    print("or AI models. You cannot use the Spotify Platform or any")
    print("Spotify Content to train ML/AI models or ingest Spotify")
    print("Content into ML/AI models.")
    print("=" * 80 + "\n")

    return results_df


if __name__ == "__main__":
    results = main()
