# Spotify API Data Enrichment

This folder contains tools to enrich song data with Spotify metadata (track info, artist info, audio features).

## Folders

### `chunks/` - OPTIMIZED (recommended)
- **Use this for production**
- Batch API calls (50x faster)
- Checkpoints every 500 songs
- Can resume if interrupted
- Artist caching to avoid duplicate lookups

### `row_by_row/` - Sequential processing
- **Use for small datasets or testing**
- Processes one song at a time
- No checkpoints
- Simpler code, easier to understand

## Quick Start

For most use cases, use the optimized version:
```cmd
cd chunks
python main_optimized.py
```

See README files in each folder for detailed usage instructions.

## Important Notice

**Spotify content may not be used to train machine learning or AI models.**
