**This directory contains scripts to generate our final training and test sets and create features for final feature selection.**



**final\_clean.ipynb:**

Creates final cleaned train and test datasets with an 85-15 split from raw dataset sourced at https://huggingface.co/datasets/ailsntua/Chordonomicon.



Cleaning steps:

1. Remove all features except for chords, decade, main\_genre and spotify\_song\_id
2. Drop songs from decades before 1950
3. Remove any data entries that have NA for any of release\_date, main\_genre and spotify\_song\_id
4. Remove any data entries that contain less than six total chords in the chords variable
5. Remove any data entries with unknown chords

Returns:
final_train: training dataset for feature engineering
final_test: test dataset



**possible\_density\_features.ipynb:**

Creates the following possible features:

* unique_chord_denstiy: number of unique chords divided by total number of chords
* unique_2gram_denstiy: number of unique 2-grams divided by total number of chords
* unique_5gram_denstiy: number of unique 5-grams divided by total number of chords







