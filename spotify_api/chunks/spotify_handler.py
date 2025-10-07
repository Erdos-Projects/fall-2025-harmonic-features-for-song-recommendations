"""
Spotify API Handler

Provides interface for interacting with the Spotify Web API using Client Credentials flow.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from typing import Optional, Dict, List
import time


class SpotifyHandler:
    """Handler class for Spotify API operations with batch processing and caching"""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize Spotify API handler

        Args:
            client_id: Spotify Client ID (reads from SPOTIPY_CLIENT_ID env var if None)
            client_secret: Spotify Client Secret (reads from SPOTIPY_CLIENT_SECRET env var if None)
        """
        try:
            if client_id and client_secret:
                self.auth_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
            else:
                # Will automatically use SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET env vars
                self.auth_manager = SpotifyClientCredentials()

            self.sp = spotipy.Spotify(
                auth_manager=self.auth_manager,
                requests_timeout=10,
                retries=3,
                status_retries=3,
                backoff_factor=0.3,
            )
            self._test_connection()
            print("Spotify API connection successful")
            
            # Add cache for artist info to avoid duplicate lookups
            self.artist_cache = {}

        except Exception as e:
            raise Exception(
                f"Failed to initialize Spotify API. "
                f"Make sure SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET are set. "
                f"Error: {e}"
            )

    def _test_connection(self):
        """Test the API connection with a simple call"""
        try:
            # Test with a known track ID
            self.sp.track("3n3Ppam7vgaVa1iaRUc9Lp")
        except Exception as e:
            raise Exception(f"Connection test failed: {e}")

    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed information about a track

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary containing track information or None if failed
        """
        try:
            track = self.sp.track(track_id)
            return {
                'track_id': track_id,
                'track_name': track['name'],
                'artists': [artist['name'] for artist in track['artists']],
                'artist_ids': [artist['id'] for artist in track['artists']],
                'album_name': track['album']['name'],
                'album_id': track['album']['id'],
                'release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'explicit': track['explicit'],
                'preview_url': track.get('preview_url'),
            }
        except SpotifyException as e:
            return None
        except Exception as e:
            return None

    def get_artist_info(self, artist_id: str) -> Optional[Dict]:
        """
        Get detailed information about an artist (with caching)

        Args:
            artist_id: Spotify artist ID

        Returns:
            Dictionary containing artist information or None if failed
        """
        # Check cache first
        if artist_id in self.artist_cache:
            return self.artist_cache[artist_id]
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                artist = self.sp.artist(artist_id)
                result = {
                    'artist_id': artist_id,
                    'artist_name': artist['name'],
                    'genres': artist.get('genres', []),
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total'],
                }
                # Cache the result
                self.artist_cache[artist_id] = result
                return result
            except SpotifyException as e:
                if hasattr(e, 'http_status') and e.http_status == 429:
                    # Rate limit hit
                    retry_after = int(e.headers.get('Retry-After', 5))
                    print(f"    ⚠️  Rate limit on artist fetch! Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    if attempt < max_retries - 1:
                        continue
                return None
            except Exception as e:
                return None

        return None

    def get_track_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a track (tempo, key, mode, etc.)

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary containing audio features or None if failed
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Ensure we have a fresh token for audio_features
                if attempt > 0:
                    # Force token refresh on retry
                    self.auth_manager.get_access_token(force_refresh=True)
                    # Recreate Spotify client with fresh token
                    self.sp = spotipy.Spotify(
                        auth_manager=self.auth_manager,
                        requests_timeout=10,
                        retries=3,
                        status_retries=3,
                        backoff_factor=0.3,
                    )

                # Note: audio_features expects a list input
                feats_list = self.sp.audio_features([track_id])

                if not feats_list:
                    return None

                feat = feats_list[0]
                if feat is None:
                    return None

                return {
                    'track_id': track_id,
                    'danceability': feat['danceability'],
                    'energy': feat['energy'],
                    'key': feat['key'],
                    'loudness': feat['loudness'],
                    'mode': feat['mode'],
                    'speechiness': feat['speechiness'],
                    'acousticness': feat['acousticness'],
                    'instrumentalness': feat['instrumentalness'],
                    'liveness': feat['liveness'],
                    'valence': feat['valence'],
                    'tempo': feat['tempo'],
                    'time_signature': feat['time_signature'],
                }
            except SpotifyException as e:
                # Check if it's a 401 error
                if hasattr(e, 'http_status') and e.http_status == 401:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                return None
            except Exception as e:
                return None

        return None

    def get_complete_track_data(self, track_id: str, artist_id: Optional[str] = None) -> Dict:
        """
        Get complete track data including track info, artist info, and audio features

        Args:
            track_id: Spotify track ID
            artist_id: Spotify artist ID (optional, will use first artist from track if not provided)

        Returns:
            Dictionary containing all available information
        """
        result = {
            'track_id': track_id,
            'artist_id': artist_id,
            'track_info': None,
            'artist_info': None,
            'audio_features': None,
            'success': False
        }

        # Get track info
        track_info = self.get_track_info(track_id)
        if track_info:
            result['track_info'] = track_info

            # If artist_id not provided, use the first artist from track
            if not artist_id and track_info['artist_ids']:
                artist_id = track_info['artist_ids'][0]
                result['artist_id'] = artist_id

        # Get artist info
        if artist_id:
            artist_info = self.get_artist_info(artist_id)
            result['artist_info'] = artist_info

        # Get audio features
        audio_features = self.get_track_audio_features(track_id)
        result['audio_features'] = audio_features

        # Mark as successful if we got at least track info
        result['success'] = track_info is not None

        return result

    def batch_get_tracks(self, track_ids: List[str], delay: float = 0.1) -> List[Dict]:
        """
        Get information for multiple tracks with rate limiting

        Args:
            track_ids: List of Spotify track IDs
            delay: Delay between requests in seconds (default 0.1)

        Returns:
            List of dictionaries containing track information
        """
        results = []
        for i, track_id in enumerate(track_ids):
            track_info = self.get_track_info(track_id)
            results.append(track_info)

            # Rate limiting
            if i < len(track_ids) - 1:
                time.sleep(delay)

        return results

    def batch_get_tracks_optimized(self, track_ids: List[str]) -> List[Optional[Dict]]:
        """
        Get information for multiple tracks using Spotify's batch endpoint
        Up to 50 tracks at a time

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            List of dictionaries containing track information
        """
        results = []
        batch_size = 50
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    tracks = self.sp.tracks(batch)
                    for track in tracks['tracks']:
                        if track is None:
                            results.append(None)
                            continue

                        results.append({
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'artists': [artist['name'] for artist in track['artists']],
                            'artist_ids': [artist['id'] for artist in track['artists']],
                            'album_name': track['album']['name'],
                            'album_id': track['album']['id'],
                            'release_date': track['album']['release_date'],
                            'duration_ms': track['duration_ms'],
                            'popularity': track['popularity'],
                            'explicit': track['explicit'],
                            'preview_url': track.get('preview_url'),
                        })
                    time.sleep(0.1)  # Small delay between batches
                    break  # Success, exit retry loop

                except SpotifyException as e:
                    if hasattr(e, 'http_status') and e.http_status == 429:
                        # Rate limit hit
                        retry_after = int(e.headers.get('Retry-After', 5))
                        print(f"    ⚠️  Rate limit hit! Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        if attempt < max_retries - 1:
                            continue
                    # If other error or max retries, append None for batch
                    results.extend([None] * len(batch))
                    break
                except Exception as e:
                    print(f"    ⚠️  Batch error: {e}")
                    # If batch fails, append None for each track in batch
                    results.extend([None] * len(batch))
                    break

        return results

    def batch_get_audio_features_optimized(self, track_ids: List[str]) -> List[Optional[Dict]]:
        """
        Get audio features for multiple tracks using Spotify's batch endpoint
        Up to 100 tracks at a time

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            List of dictionaries containing audio features
        """
        results = []
        batch_size = 100
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        self.auth_manager.get_access_token(force_refresh=True)
                        self.sp = spotipy.Spotify(
                            auth_manager=self.auth_manager,
                            requests_timeout=10,
                            retries=3,
                            status_retries=3,
                            backoff_factor=0.3,
                        )
                    
                    features_list = self.sp.audio_features(batch)
                    
                    for feat in features_list:
                        if feat is None:
                            results.append(None)
                            continue
                        
                        results.append({
                            'track_id': feat['id'],
                            'danceability': feat['danceability'],
                            'energy': feat['energy'],
                            'key': feat['key'],
                            'loudness': feat['loudness'],
                            'mode': feat['mode'],
                            'speechiness': feat['speechiness'],
                            'acousticness': feat['acousticness'],
                            'instrumentalness': feat['instrumentalness'],
                            'liveness': feat['liveness'],
                            'valence': feat['valence'],
                            'tempo': feat['tempo'],
                            'time_signature': feat['time_signature'],
                        })
                    
                    time.sleep(0.1)  # Small delay between batches
                    break
                    
                except SpotifyException as e:
                    if hasattr(e, 'http_status') and e.http_status == 401:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                    # If we failed, append None for each track in batch
                    results.extend([None] * len(batch))
                    break
                except Exception as e:
                    results.extend([None] * len(batch))
                    break
                    
        return results

    def clear_artist_cache(self):
        """Clear the artist cache"""
        self.artist_cache = {}
        
    def get_artist_cache_stats(self):
        """Get statistics about the artist cache"""
        return {
            'cached_artists': len(self.artist_cache),
            'cache_keys': list(self.artist_cache.keys())[:10]  # First 10 for preview
        }
