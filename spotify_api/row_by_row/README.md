# Spotify API Data Fetcher

## Setup

1. Get Spotify credentials at https://developer.spotify.com/dashboard
2. Create an app and copy your Client ID and Client Secret
3. Set environment variables:

    ```cmd
    set SPOTIPY_CLIENT_ID=your_client_id
    set SPOTIPY_CLIENT_SECRET=your_client_secret
    ```

## Usage

**Run with defaults:**

    ```cmd
    python main.py
    ```

**Configure in `main.py`:**

    ```python
    # File paths
    input_file = "chordonomicon_part_2.csv"
    output_file = "chordonomicon_part_2_enriched.csv"

    # What to fetch
    get_track_info = True       # Track name, album, popularity, etc.
    get_artist_info = True      # Artist name, genres, followers
    get_audio_features = False  # Tempo, key, danceability, etc.

    # Processing
    num_rows = 10               # None = all rows, or set number for testing
    ```

**Testing vs Production:**
- **Testing**: Set `num_rows = 10` to test without processing entire dataset
- **Production**: Set `num_rows = None` to process all rows

## Input Format

CSV with columns:
- `spotify_song_id` - Spotify track ID (required)
- `spotify_artist_id` - Spotify artist ID (optional)

## Output

Single CSV file with enriched data including track information, artist details, and optionally audio features.

## Important Notice

**Spotify content may not be used to train machine learning or AI models.**

## Spotipy `.cache` File

- Spotipy creates a `.cache` file to store authentication tokens.
- **Do not commit `.cache` to version control.**  
  Add `.cache` to your `.gitignore` to protect your API credentials.
- You can change the cache location by setting the `cache_path` parameter in Spotipy.

Example for `.gitignore`:
```
.cache
```
Example for custom cache path in Python:
```python
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(cache_path="your_custom_cache_path"))
```
