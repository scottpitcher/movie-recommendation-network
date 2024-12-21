# api_fetch.py
import requests
import pandas as pd
import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Base TMDb URL
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movies(api_key, language="en-US", total_pages=100):
    """
    Fetch popular movies from TMDb and return them as a pandas DataFrame.
    """
    logger.info(f"Fetching movie data for {total_pages} pages...")
    endpoint = f"{BASE_URL}/movie/popular"
    all_movies = []
    
    for page in range(1, total_pages + 1):
        params = {
            "api_key": api_key,
            "language": language,
            "page": page
        }
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched page {page} with {len(data['results'])} movies.")

        movies = [
            {
                "id": movie["id"],
                "title": movie["title"],
                "overview": movie["overview"],
                "popularity": movie["popularity"],
                "release_date": movie["release_date"],
                "vote_average": movie["vote_average"],
                "vote_count": movie["vote_count"],
                "genre_ids": movie["genre_ids"]
            }
            for movie in data["results"]
        ]
        all_movies.extend(movies)
    
    logger.info("Successfully fetched all movie data.")
    return pd.DataFrame(all_movies)

def fetch_movie_credits(api_key, movie_id):
    """
    Fetch cast and crew (actors and directors) for a given movie from TMDb.
    """
    endpoint = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": api_key}

    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        logger.warning(f"Failed to fetch credits for movie ID {movie_id}: {response.status_code}")
        return {"actors": [], "director": None}

    data = response.json()
    
    # Extract top 15 actors
    actors = [cast["name"] for cast in data.get("cast", [])[:15]]
    
    # Extract director
    directors = [crew["name"] for crew in data.get("crew", []) if crew["job"] == "Director"]
    director = directors[0] if directors else None
    
    return {"actors": actors, "director": director}

def fetch_genres(api_key):
    """
    Fetch genre IDs and their corresponding names from TMDb.
    """
    logger.info("Fetching genre mappings from TMDb...")
    endpoint = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": api_key, "language": "en-US"}

    response = requests.get(endpoint, params=params)
    response.raise_for_status()

    genres = response.json().get("genres", [])
    genre_mapping = {genre["id"]: genre["name"] for genre in genres}
    logger.info("Successfully fetched genre mappings.")
    return genre_mapping

def process_movie_data(api_key, total_pages=200, output_path="data/full_data_with_credits.csv"):
    """
    Fetch and process movie data, including genres, actors, and directors.
    """
    # Fetch movies
    dataset = fetch_movies(api_key, total_pages=total_pages)
    
    # Fetch genres
    genre_mapping = fetch_genres(api_key)
    dataset['genres'] = dataset['genre_ids'].apply(lambda ids: [genre_mapping.get(genre) for genre in ids])
    
    # Initialize columns for actors and directors
    dataset["actors"] = ""
    dataset["director"] = ""

    # Fetch credits for each movie
    logger.info("Fetching credits for each movie...")
    for i, (idx, row) in enumerate(dataset.iterrows()):
        credits = fetch_movie_credits(api_key, row["id"])
        dataset.at[idx, "actors"] = credits["actors"]
        dataset.at[idx, "director"] = credits["director"]
        
        # Rate limiting
        time.sleep(0.5)  # Adjust to avoid hitting API rate limits
        
        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i + 1} movies...")

    # Save the dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to '{output_path}'.")

if __name__ == "__main__":
    # Fetch API key from .env file
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        logger.error("API key is missing. Please set it in the .env file.")
        exit(1)
    
    # Fetch and process movie data
    process_movie_data(API_KEY, total_pages=200, output_path="data/raw_data/full_data_with_credits.csv")
