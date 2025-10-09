# Spotify API Data Fetcher - OPTIMIZED

## Key Features

- **Batch API calls**: 50 tracks + 100 audio features at once (~50x faster)
- **Artist caching**: Eliminates duplicate lookups across chunks
- **Checkpoints**: Saves every 500 songs (default), can resume if interrupted
- **Performance**: ~20-50 songs/sec vs ~1-2 songs/sec (old pipeline)

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
python main_optimized.py
```

**Configure in `main_optimized.py`:**
```python
# File paths
input_file = "chordonomicon_part_2.csv"
output_dir = "output_spotify"              # Folder for checkpoint files
output_dir_final = "output_spotify_final"  # Folder for final combined file

# What to fetch
get_track_info = True       # Track name, album, popularity, etc.
get_artist_info = True      # Artist name, genres, followers
get_audio_features = False  # Tempo, key, danceability, etc.

# Processing
num_rows = None             # None = all rows, or set number for testing
chunk_size = 500            # Songs per checkpoint
resume = True               # Continue from last checkpoint
```

**Testing vs Production:**
- **Testing**: Set `num_rows = 50` to test without processing entire dataset
- **Production**: Set `num_rows = None` to process all rows
- **Chunk size**: Keep default 500 (smaller = faster checkpoints, larger = faster overall)

## Input Format

CSV with columns:
- `spotify_song_id` - Spotify track ID (required)
- `spotify_artist_id` - Spotify artist ID (optional)

## Output

Creates checkpoint files in one folder and final combined file in another:
```
output_spotify/                                      (checkpoints)
  ├── chordonomicon_part_2_enriched_chunk_000.csv
  ├── chordonomicon_part_2_enriched_chunk_001.csv
  └── chordonomicon_part_2_enriched_chunk_002.csv

output_spotify_final/                                (final output)
  └── chordonomicon_part_2_enriched_final.csv       (use this)
```

**To resume after interruption:** Just run `python main_optimized.py` again

## Troubleshooting

**Rate limits**: Script auto-retries with backoff  
**Memory issues**: Reduce `chunk_size` to 100-250  
**Missing data**: Check `success` column for failed rows

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
