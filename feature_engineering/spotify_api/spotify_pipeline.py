"""
Spotify API Pipeline - Optimized Batch Processing

Processes data in chunks with batch API calls and artist caching for improved performance.

Features:
- Batch processing (50 tracks, 100 audio features at once)
- Artist caching to eliminate duplicate lookups
- Checkpoint saves after each chunk
- Resume capability
"""

import pandas as pd
from pathlib import Path
from spotify_handler import SpotifyHandler
import time
import os


def spotify_enrichment_pipeline(
    input_file,
    output_dir="output",
    output_dir_final="output_final",
    base_output_name="spotify_enriched_data",
    get_track_info=True,
    get_artist_info=True,
    get_audio_features=True,
    num_rows=None,
    chunk_size=500,
    resume=True
):
    """
    Pipeline to fetch Spotify data and enrich CSV

    Args:
        input_file: Path to input CSV with 'spotify_song_id' column
        output_dir: Directory for checkpoint files
        output_dir_final: Directory for final combined file
        base_output_name: Base name for output files
        get_track_info: Fetch track information (default: True)
        get_artist_info: Fetch artist information (default: True)
        get_audio_features: Fetch audio features (default: True)
        num_rows: Number of rows to process (None = all, default: None)
        chunk_size: Songs per checkpoint (default: 500)
        resume: Resume from last checkpoint (default: True)

    Returns:
        DataFrame with enriched Spotify data
    """

    print("=" * 80)
    print("Spotify Track Information Fetcher - OPTIMIZED Pipeline")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory (checkpoints): {output_dir}")
    print(f"Output directory (final): {output_dir_final}")
    print(f"Chunk size: {chunk_size:,} songs")
    print(f"Fetching: Track={get_track_info}, Artist={get_artist_info}, Audio={get_audio_features}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_final, exist_ok=True)

    # Initialize Spotify handler
    print("\nInitializing Spotify API connection...")
    try:
        spotify = SpotifyHandler()
    except Exception as e:
        print(f"\nFailed to initialize Spotify API: {e}")
        print("\nTo set up credentials:")
        print("   1. Go to https://developer.spotify.com/dashboard")
        print("   2. Create an app and get Client ID and Client Secret")
        print("   3. Set environment variables:")
        print("      set SPOTIPY_CLIENT_ID=your_client_id")
        print("      set SPOTIPY_CLIENT_SECRET=your_client_secret")
        return None

    # Read CSV file
    print(f"\nReading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"Total rows in file: {len(df):,}")
    except Exception as e:
        print(f"\nError reading CSV file: {e}")
        return None

    # Limit rows if specified
    if num_rows is not None:
        print(f"Limiting to first {num_rows:,} rows...")
        df = df.head(num_rows)

    total_rows = len(df)
    print(f"Processing {total_rows:,} total rows in chunks of {chunk_size:,}")

    # Check for existing checkpoint
    start_chunk = 0
    all_results = []

    if resume:
        checkpoint_pattern = f"{base_output_name}_chunk_*.csv"
        existing_files = sorted(Path(output_dir).glob(checkpoint_pattern))
        if existing_files:
            last_file = existing_files[-1]
            # Extract chunk number from filename
            last_chunk_num = int(last_file.stem.split('_chunk_')[1])
            start_chunk = last_chunk_num + 1
            print(f"\nResuming from chunk {start_chunk} (found {len(existing_files)} existing chunks)")

            # Load existing results
            for f in existing_files:
                chunk_df = pd.read_csv(f)
                all_results.append(chunk_df)
                print(f"  Loaded {f.name}: {len(chunk_df):,} rows")

    # Calculate chunks
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    # Start timing
    overall_start = time.time()

    # Process each chunk
    for chunk_idx in range(start_chunk, num_chunks):
        chunk_start_idx = chunk_idx * chunk_size
        chunk_end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
        chunk_df = df.iloc[chunk_start_idx:chunk_end_idx].copy()

        print(f"\n{'=' * 80}")
        print(f"Processing Chunk {chunk_idx + 1}/{num_chunks}")
        print(f"Rows {chunk_start_idx:,} to {chunk_end_idx:,} ({len(chunk_df):,} songs)")
        print(f"{'=' * 80}")

        chunk_start_time = time.time()

        # Get track IDs
        track_ids = chunk_df['spotify_song_id'].tolist()
        print(f"  Extracted {len(track_ids)} track IDs from chunk")

        # OPTIMIZATION: Batch fetch track info
        if get_track_info:
            print(f"\nFetching track info in batches of 50...")
            print(f"  Starting batch fetch at {time.strftime('%H:%M:%S')}")
            try:
                track_infos = spotify.batch_get_tracks_optimized(track_ids)
                print(f"  ✓ Retrieved {sum(1 for t in track_infos if t is not None)}/{len(track_ids)} tracks")
            except Exception as e:
                print(f"  ✗ ERROR fetching track info: {e}")
                track_infos = [None] * len(track_ids)
        else:
            print(f"\nSkipping track info (disabled)")
            track_infos = [None] * len(track_ids)

        # OPTIMIZATION: Batch fetch audio features
        if get_audio_features:
            print(f"\nFetching audio features in batches of 100...")
            print(f"  Starting batch fetch at {time.strftime('%H:%M:%S')}")
            try:
                audio_features = spotify.batch_get_audio_features_optimized(track_ids)
                print(f"  ✓ Retrieved {sum(1 for a in audio_features if a is not None)}/{len(track_ids)} audio features")
            except Exception as e:
                print(f"  ✗ ERROR fetching audio features: {e}")
                audio_features = [None] * len(track_ids)
        else:
            print(f"\nSkipping audio features (disabled)")
            audio_features = [None] * len(track_ids)

        # OPTIMIZATION: Collect unique artist IDs and fetch them (with caching)
        if get_artist_info:
            print(f"\nCollecting unique artist IDs...")
            # Get all unique artist IDs from track info
            unique_artist_ids = set()
            for track_info in track_infos:
                if track_info and track_info.get('artist_ids'):
                    artist_id = track_info['artist_ids'][0]
                    # Only add valid artist IDs
                    if artist_id and isinstance(artist_id, str) and len(artist_id) > 0:
                        unique_artist_ids.add(artist_id)

            # Also add artist IDs from input data
            for artist_id in chunk_df['spotify_artist_id'].dropna():
                # Filter out empty strings and ensure it's a valid string
                if artist_id and isinstance(artist_id, str) and len(artist_id) > 0 and artist_id.lower() != 'nan':
                    unique_artist_ids.add(artist_id)

            print(f"  Found {len(unique_artist_ids)} unique artists to fetch")
            print(f"  Current artist cache has {len(spotify.artist_cache)} entries")
            print(f"  Starting artist fetch at {time.strftime('%H:%M:%S')}")

            # Fetch artist info (will use cache automatically)
            artists_fetched = 0
            artists_from_cache = 0
            for artist_id in unique_artist_ids:
                if artist_id not in spotify.artist_cache:
                    spotify.get_artist_info(artist_id)
                    artists_fetched += 1
                    if artists_fetched % 100 == 0:
                        print(f"  → Progress: Fetched {artists_fetched} new artists...")
                    elif artists_fetched % 50 == 0:
                        print(f"  → {artists_fetched} artists fetched...", end='\r')
                    time.sleep(0.05)  # Small delay
                else:
                    artists_from_cache += 1

            print(f"  ✓ Fetched {artists_fetched} new artists, {artists_from_cache} from cache")
        else:
            print(f"\nSkipping artist info (disabled)")

        # Build results DataFrame for this chunk
        print(f"\nBuilding results DataFrame for chunk {chunk_idx + 1}...")
        print(f"  Processing {len(chunk_df)} rows at {time.strftime('%H:%M:%S')}")
        summary_data = []

        for i, (idx, row) in enumerate(chunk_df.iterrows()):
            # Progress indicator every 1000 rows
            if (i + 1) % 1000 == 0:
                print(f"  → Processing row {i + 1}/{len(chunk_df)}...", end='\r')

            track_id = row['spotify_song_id']
            artist_id = row['spotify_artist_id']
            track_info = track_infos[i]
            audio_feat = audio_features[i]

            row_data = {
                'track_id': track_id,
                'artist_id': artist_id,
                'success': track_info is not None
            }

            # Add track info
            if get_track_info and track_info:
                row_data.update({
                    'track_name': track_info['track_name'],
                    'artists': ', '.join(track_info['artists']),
                    'album_name': track_info['album_name'],
                    'release_date': track_info['release_date'],
                    'popularity': track_info['popularity'],
                    'duration_ms': track_info['duration_ms'],
                })

                # Update artist_id if not provided
                if not artist_id and track_info['artist_ids']:
                    artist_id = track_info['artist_ids'][0]
                    row_data['artist_id'] = artist_id

            # Add artist info from cache
            if get_artist_info and artist_id:
                artist_info = spotify.get_artist_info(artist_id)  # Uses cache
                if artist_info:
                    row_data.update({
                        'artist_name': artist_info['artist_name'],
                        'genres': ', '.join(artist_info['genres']) if artist_info['genres'] else '',
                        'artist_popularity': artist_info['popularity'],
                        'followers': artist_info['followers'],
                    })

            # Add audio features
            if get_audio_features and audio_feat:
                row_data.update({
                    'key': audio_feat['key'],
                    'mode': audio_feat['mode'],
                    'tempo': audio_feat['tempo'],
                    'time_signature': audio_feat['time_signature'],
                    'danceability': audio_feat['danceability'],
                    'energy': audio_feat['energy'],
                    'valence': audio_feat['valence'],
                    'acousticness': audio_feat['acousticness'],
                    'instrumentalness': audio_feat['instrumentalness'],
                    'speechiness': audio_feat['speechiness'],
                    'liveness': audio_feat['liveness'],
                    'loudness': audio_feat['loudness'],
                })

            summary_data.append(row_data)

        print(f"  ✓ Completed processing all {len(chunk_df)} rows")
        print(f"\nCreating DataFrame and saving to file...")
        chunk_results_df = pd.DataFrame(summary_data)

        # Save chunk
        chunk_filename = f"{base_output_name}_chunk_{chunk_idx:03d}.csv"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        print(f"  Writing to: {chunk_filename}")
        chunk_results_df.to_csv(chunk_filepath, index=False)
        print(f"  ✓ File saved successfully")

        chunk_elapsed = time.time() - chunk_start_time
        successful = chunk_results_df['success'].sum()

        print(f"\n{'=' * 80}")
        print(f"Chunk {chunk_idx + 1} Summary:")
        print(f"  Successfully processed: {successful}/{len(chunk_results_df)} rows")
        print(f"  Time: {chunk_elapsed:.2f}s ({chunk_elapsed/60:.2f} min)")
        print(f"  Rate: {len(chunk_results_df)/chunk_elapsed:.1f} songs/sec")
        print(f"  Saved to: {chunk_filename}")
        print(f"  Artist cache size: {len(spotify.artist_cache)} artists")
        print(f"{'=' * 80}")

        all_results.append(chunk_results_df)

    # Combine all chunks
    print(f"\n{'=' * 80}")
    print("Combining all chunks into final output...")
    print(f"{'=' * 80}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Save final combined file
        final_filename = f"{base_output_name}_final.csv"
        final_filepath = os.path.join(output_dir_final, final_filename)
        final_df.to_csv(final_filepath, index=False)

        overall_elapsed = time.time() - overall_start
        total_successful = final_df['success'].sum()

        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total rows processed: {len(final_df):,}")
        print(f"Successfully processed: {total_successful:,}/{len(final_df):,}")
        print(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} min)")
        print(f"Average rate: {len(final_df)/overall_elapsed:.1f} songs/sec")
        print(f"Total columns: {len(final_df.columns)}")
        print(f"Artist cache utilized: {len(spotify.artist_cache)} unique artists")
        print(f"\nFinal output saved to: {final_filename}")
        print(f"{'=' * 80}\n")

        return final_df
    else:
        print("No results to combine.")
        return None
