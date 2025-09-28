Summary of data cleaning steps:

Data entries and features removed:
1)Remove any data entries that have NA for all three of release_date, main_genre and spotify_song_id
2) Drop the genre and rock_genre features since they are too specific to be useful
3) Drop the id feature since it contains no useful information
4) Remove any data entries that contain less than three total chords in the chords variable

New features:
chord_dict: modifies the chords feature to turn it into a dictionary format where the keys are the sections (verse, chorus etc.) by using Iliyana's sandbox code
num_sections: counts the total number of sections (verse, chorus, etc.) listed in the chords feature 
tot_chords: total number of chord symbols listed in the chords feature
tot_unique_chords: total number of unique chord symbols listed in the chords feature
unique_chord_density: total_unique_chords/tot_chords
 