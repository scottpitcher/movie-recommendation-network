import requests
import pandas as pd

def fetch_movies(api_key, language="en-US", page=1):
    """
    Fetch popular movies from TMDb and return them as a pandas DataFrame.
    
    Parameters:
    - api_key (str): Your TMDb API key.
    - language (str): Language for the movie data (default: "en-US").
    - page (int): Page number for paginated results (default: 1).
    
    Returns:
    - pd.DataFrame: DataFrame containing movie details.
    """
    BASE_URL = "https://api.themoviedb.org/3"
    endpoint = f"{BASE_URL}/movie/popular"
    
    params = {
        "api_key": api_key,
        "language": language,
        "page": page
    }
    
    # Make the API request
    response = requests.get(endpoint, params=params)
    response.raise_for_status()  # Raise an error for bad responses (e.g., 401, 404)
    
    # Parse JSON data
    data = response.json()
    
    # Extract relevant fields from results
    movies = [
        {
            "id": movie["id"],
            "title": movie["title"],
            "release_date": movie["release_date"],
            "vote_average": movie["vote_average"],
            "genre_ids": movie["genre_ids"]  # Genre IDs (can map to actual genre names later)
        }
        for movie in data["results"]
    ]
    
    # Convert to pandas DataFrame
    return pd.DataFrame(movies)

# Example usage:
if __name__ == "__main__":
    API_KEY = "your_api_key_here"  # Replace with your actual TMDb API key
    dataset = fetch_movies(API_KEY)
    print(dataset.head())
