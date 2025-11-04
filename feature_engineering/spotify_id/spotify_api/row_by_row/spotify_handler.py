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
    """Handler class for Spotify API operations"""

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
        Get detailed information about an artist

        Args:
            artist_id: Spotify artist ID

        Returns:
            Dictionary containing artist information or None if failed
        """
        try:
            artist = self.sp.artist(artist_id)
            return {
                'artist_id': artist_id,
                'artist_name': artist['name'],
                'genres': artist.get('genres', []),
                'popularity': artist['popularity'],
                'followers': artist['followers']['total'],
            }
        except SpotifyException as e:
            return None
        except Exception as e:
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
