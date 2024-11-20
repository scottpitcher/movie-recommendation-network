# api_fetch.py
import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
load_dotenv()

def fetch_movies(api_key, language="en-US", total_pages=100):
    """
    Fetch popular movies from TMDb and return them as a pandas DataFrame.
    
    Parameters:
    - api_key (str): Your TMDb API key.
    - language (str): Language for the movie data (default: "en-US").
    - total_pages (int): Number of pages to fetch (default: 5).
    
    Returns:
    - pd.DataFrame: DataFrame containing movie details.
    """
    BASE_URL = "https://api.themoviedb.org/3"
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
        print(f'Fetching page {page}: {len(data["results"])} movies')

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
    
    return pd.DataFrame(all_movies)

def fetch_movie_credits(api_key, movie_id):
    """
    Fetch cast and crew (actors and directors) for a given movie from TMDb.
    
    Parameters:
    - api_key (str): Your TMDb API key.
    - movie_id (int): The ID of the movie.
    
    Returns:
    - dict: Dictionary containing the top actors and director.
    """
    BASE_URL = "https://api.themoviedb.org/3"
    endpoint = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": api_key}

    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch credits for movie ID {movie_id}: {response.status_code}")
        return {"actors": [], "director": None}
    
    data = response.json()
    
    # Extract top 5 actors
    actors = [cast["name"] for cast in data.get("cast", [])[:5]]
    
    # Extract director
    directors = [crew["name"] for crew in data.get("crew", []) if crew["job"] == "Director"]
    director = directors[0] if directors else None
    
    return {"actors": actors, "director": director}


def fetch_genres():
    """
    Fetch genre IDs and their corresponding names from TMDb.
    
    Returns:
    - dict: A dictionary mapping genre IDs to genre names.
    """
    url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
    
    # Add your Bearer token here
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyZjliMGMwNTZiOWMxMGM0YTVjMTJiNmI2MjBjZDEyZCIsIm5iZiI6MTczMTk0OTE3Mi4zMjIzMzA3LCJzdWIiOiI2NzNhMTY3NTJjMGI3ZmQyMDM0YTk4NzAiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.-CpSsrjp68cZy7DJamUx4sah_oQj7aFlL4qXkTSbfeM",
        "accept": "application/json"
    }
    
    # Make the API request
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request fails
    
    # Parse the JSON response
    genres = response.json().get("genres", [])
    
    # Convert to a dictionary for easy lookup
    genre_mapping = {genre["id"]: genre["name"] for genre in genres}
    
    return genre_mapping


if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY")
    
    # Fetch movies
    dataset = fetch_movies(API_KEY, total_pages=100)
    
    # Fetch genres
    GENRE_MAPPING = fetch_genres()
    dataset['genres'] = dataset['genre_ids'].apply(lambda ids: [GENRE_MAPPING.get(genre) for genre in ids])
    
    # Initialize columns for actors and directors
    dataset["actors"] = ""
    dataset["director"] = ""

    # Fetch credits for each movie
    print("\nFetching credits for each movie...")
    for i, (idx, row) in enumerate(dataset.iterrows()):
        credits = fetch_movie_credits(API_KEY, row["id"])
        dataset.at[idx, "actors"] = credits["actors"]
        dataset.at[idx, "director"] = credits["director"]
        
        # Rate limiting
        time.sleep(0.5)  # Adjust to avoid hitting API rate limits

        if i%20 ==0 and i !=0:
            print(f"Processed page {int(i/20)}")

    # Save the dataset
    dataset.to_csv('data/full_data_with_credits.csv', index=False)
    print("Dataset saved to 'data/full_data_with_credits.csv'")
