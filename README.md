#                                                                             <p align="center"> Harmonic Features for Song Recommendations </p>

[Google Drive for raw datasets](https://drive.google.com/drive/folders/1wQqr2Wh-QhNbvWzvqfjFSGo5dHFz1YxF?usp=drive_link)

Team members: [Juan Salinas](https://github.com/juansalinas2), [Iliyana Dobreva](https://github.com/iliyanadobreva), [Joshua Ruiter](https://github.com/JoshuaRuiter), [Matthew Dykes](https://github.com/mmd266-svg), [Elizabeth Rizor](https://github.com/ejrise)

#  Table of Contents</p>

1. [Introduction](#introduction)  
2. [Dataset Curation](#dataset-curation)  
3. [Data Cleaning and EDA](#data-cleaning-and-EDA)  
4. [Feature Engineering](#feature-engineering) 
5. [Feature Selection](#feature-selection) 
5. [Modeling](#modeling)
5. [Results](#results)  
6. [Conclusion](#conclusion)  
7. [Description of Repository](#description-of-repository)  

## Introduction
&nbsp;&nbsp;&nbsp;&nbsp; Music has served as an important medium for human expression for tens of thousands of years. Songs can be broken down into quantifiable harmonic patterns that can signal a cultural moment in time or belonging to a specific subculture. Mapping these harmonic features to markers of cultural relevance are useful for musicians, researchers, and businesses (such as streaming platforms) who wish to better recommend songs to individuals. Because of this need, we aimed to explore how harmony shapes the sound of popular music, and determine whether harmonic "fingerprints" can accurately classify songs by genre, decade, and popularity. 

## Dataset Curation 
&nbsp;&nbsp;&nbsp;&nbsp; Our primary dataset was the Chordonomicon dataset, which contains over 600,000 chord-annotated songs along with their release decade, genre, and Spotify ID. To predict popularity, we scraped Spotify current popularity metrics with each song's Spotify ID (when available). We also used the Billboard Hot 100 dataset (contains all songs on Billboard Hot 100 going back to its inception) to label whether a song had ever reached the top charts in the US. 
1. [Chordonomincon Dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon)
2. [Spotify API](https://developer.spotify.com/)
3. [Hot 100 Dataset](https://github.com/utdata/rwd-billboard-data/blob/main/data-out/hot-100-current.csv)

## Data Cleaning and EDA
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
The notebook to get the final cleaned dataset is [here](https://github.com/Erdos-Projects/fall-2025-harmonic-features-for-song-recommendations/tree/main/Final/final_clean.ipynb).

## Feature Engineering

## Feature Selection

## Modeling

## Results

## Conclusions

## Description of Repository



