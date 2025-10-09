"""
Spotify API Pipeline - Sequential Processing

Processes song data one at a time using the Spotify API.
"""

import pandas as pd
from spotify_handler import SpotifyHandler
import time


def spotify_enrichment_pipeline(
    input_file,
    output_file,
    get_track_info=True,
    get_artist_info=True,
    get_audio_features=True,
    num_rows=None
):
    """
    Pipeline to fetch Spotify data and enrich CSV

    Args:
        input_file: Path to input CSV with 'spotify_song_id' column
        output_file: Path to output CSV for enriched data
        get_track_info: Fetch track information (default: True)
        get_artist_info: Fetch artist information (default: True)
        get_audio_features: Fetch audio features (default: True)
        num_rows: Number of rows to process (None = all, default: None)

    Returns:
        DataFrame with enriched Spotify data
    """

    print("=" * 60)
    print("Spotify Track Information Fetcher - Pipeline")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Fetching: Track={get_track_info}, Artist={get_artist_info}, Audio={get_audio_features}")

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
        print(f"Processing first {num_rows} rows...")
        df = df.head(num_rows)
        total_rows = num_rows
    else:
        print(f"Processing all {len(df):,} rows...")
        total_rows = len(df)

    # Start timing
    start_time = time.time()

    # Process each row
    results = []
    for idx, row in df.iterrows():
        track_id = row['spotify_song_id']
        artist_id = row['spotify_artist_id']

        # Initialize result structure
        data = {
            'track_id': track_id,
            'artist_id': artist_id,
            'track_info': None,
            'artist_info': None,
            'audio_features': None,
            'success': False
        }

        # Get track info if requested
        if get_track_info:
            track_info = spotify.get_track_info(track_id)
            if track_info:
                data['track_info'] = track_info
                data['success'] = True
                # If artist_id not provided, use the first artist from track
                if not artist_id and track_info['artist_ids']:
                    artist_id = track_info['artist_ids'][0]
                    data['artist_id'] = artist_id

        # Get artist info if requested
        if get_artist_info and artist_id:
            artist_info = spotify.get_artist_info(artist_id)
            data['artist_info'] = artist_info

        # Get audio features if requested
        if get_audio_features:
            audio_features = spotify.get_track_audio_features(track_id)
            data['audio_features'] = audio_features

        # One-line status output
        status_parts = []
        status_parts.append(f"[{idx + 1}/{total_rows}]")

        if get_track_info:
            status_parts.append(f"Track: {'OK' if data['track_info'] else 'FAIL'}")
        if get_artist_info:
            status_parts.append(f"Artist: {'OK' if data['artist_info'] else 'FAIL'}")
        if get_audio_features:
            status_parts.append(f"Audio: {'OK' if data['audio_features'] else 'FAIL'}")

        # Add song name if available
        if data['track_info']:
            song_name = data['track_info']['track_name'][:30]  # Truncate if too long
            status_parts.append(f"- {song_name}")

        print(" | ".join(status_parts))

        results.append(data)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Create summary DataFrame
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    summary_data = []
    for data in results:
        row_data = {
            'track_id': data['track_id'],
            'artist_id': data['artist_id'],
            'success': data['success']
        }

        if get_track_info and data['track_info']:
            ti = data['track_info']
            row_data.update({
                'track_name': ti['track_name'],
                'artists': ', '.join(ti['artists']),
                'album_name': ti['album_name'],
                'release_date': ti['release_date'],
                'popularity': ti['popularity'],
                'duration_ms': ti['duration_ms'],
            })

        if get_artist_info and data['artist_info']:
            ai = data['artist_info']
            row_data.update({
                'artist_name': ai['artist_name'],
                'genres': ', '.join(ai['genres']) if ai['genres'] else '',
                'artist_popularity': ai['popularity'],
                'followers': ai['followers'],
            })

        if get_audio_features and data['audio_features']:
            af = data['audio_features']
            row_data.update({
                'key': af['key'],
                'mode': af['mode'],
                'tempo': af['tempo'],
                'time_signature': af['time_signature'],
                'danceability': af['danceability'],
                'energy': af['energy'],
                'valence': af['valence'],
                'acousticness': af['acousticness'],
                'instrumentalness': af['instrumentalness'],
                'speechiness': af['speechiness'],
                'liveness': af['liveness'],
                'loudness': af['loudness'],
            })

        summary_data.append(row_data)

    # Create and display summary
    summary_df = pd.DataFrame(summary_data)

    successful = summary_df['success'].sum()
    has_audio = sum(1 for data in results if data['audio_features'] is not None) if get_audio_features else 0

    print(f"\nSuccessfully processed: {successful}/{len(summary_df)} rows")
    if get_audio_features:
        print(f"With audio features: {has_audio}/{len(summary_df)} rows")
    print(f"Total columns retrieved: {len(summary_df.columns)}")

    # Display timing information
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print(f"\n{'=' * 60}")
    print("Processing completed")
    print(f"{'=' * 60}\n")

    return summary_df
