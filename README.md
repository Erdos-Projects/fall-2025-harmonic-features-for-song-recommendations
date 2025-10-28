# fall-2025-harmonic-features-for-song-recommendations
Team project: fall-2025-harmonic-features-for-song-recommendations

[Google Drive for raw datasets](https://drive.google.com/drive/folders/1wQqr2Wh-QhNbvWzvqfjFSGo5dHFz1YxF?usp=drive_link)


## EDA (Matt)

The raw Chordonomicon dataset is a 679,307 $\times$ 10 data frame, with each entry representing a song and features representing information regarding song identification, chord progression, genre, year and artist identification. Since our hypothesis is that information contained in the chord progression can be used to predict the genre, release decade or popularity of a song, we only retain three of these features as targets:
* main_genre: categorical feature with 12 possibilites
* decade: release decade from 1890s-2020s, categorical variable with 13 possibilities
* spotify_song_id: song identification string, can be merged with Spotify data to generate a popularity measure

The first data cleaning step is to remove all other features not involving chord progression information. We also remove any song entries which have n/a recorded for any one of our target variables, which removes half our data entires, but this is acceptable since our data frame is so large to begin with. Next, we take a look at the distribution of the main_genre and decade features:

<p align="center">
  <img src="data/decades.png" width="600" alt="Logo" />
</p>

<p align="center">
  <img src="data/genres.png" width="600" alt="Logo" />
</p>

We see extreme class imbalance for all decades prior to the 1950s, which have class counts less than 1/1000 of the largest class, so we focus our analysis on decades beyond the 1940s. Main_genre also shows imbalance, but we retail all classes because the smallest classes are still well within a factor of 50 of the largest class and have counts in the thousands. 

We find that a small portion of the resulting data frame (about 0.2%) consists of songs whose chord progressions contain fewer than 6 total chords (i.e. fewer than 5 chord changes). Since we plan on test chord progression-related features which are functions of sequences of up to 5 chords, we also remove these entries from the data frame. 

Our final data frame is size 300,713 $\times$ (3+ # of chord progression-derived predictors)