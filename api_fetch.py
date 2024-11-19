# api_fetch.py
import requests
import pandas as pd
import os
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
    dataset = fetch_movies(API_KEY)
    GENRE_MAPPING = fetch_genres()
    dataset['genres'] = dataset['genre_ids'].apply(lambda ids: [GENRE_MAPPING.get(genre) for genre in ids])
    print(dataset.head())
    dataset.to_csv('data/full_data.csv')
