import pandas as pd

# Load the final dataset
df = pd.read_csv("data/training_data/final_dataset0.csv")

# STEP 1: Extract the data for the dropdown (InputMovie)
input_movie_dropdown = df[['InputMovie_id', 'InputMovie_title']].drop_duplicates()

# STEP 2: Extract the necessary features for the model
candidate_movie_data = df[['InputMovie_id', 'InputMovie_title', 'CandidateMovie_id', 'CandidateMovie_title',
                           'normalized_CandidateMovie_vote_count', 'normalized_CandidateMovie_log_popularity',
                           'normalized_shared_genres', 'normalized_shared_actors', 'normalized_shared_directors']]

# STEP 3: Save the processed data for future use (optional, if you want to load it later)
candidate_movie_data.to_csv("data/processed_data/candidate_movie_data.csv", index=False)

# Optionally save the input movie data if needed
input_movie_dropdown.to_csv("data/processed_data/input_movie_dropdown.csv", index=False)

print("Data processed and saved.")
