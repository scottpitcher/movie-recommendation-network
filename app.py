from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import torch
import pandas as pd
from src.model_architecture import MovieRankingModel
from fastapi import Request

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (allow requests from your portfolio site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://scottpitcher.github.io"],  # Replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Load the model
model_path = "models/movie_ranking_model.pt"
movie_feature_dim = 2  # Number of movie-level features
pairwise_feature_dim = 3  # Number of pairwise features

# Initialize the model
model = MovieRankingModel(movie_feature_dim, pairwise_feature_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the preprocessed data
input_movie_dropdown = pd.read_csv("data/deployment_data/input_movie_dropdown.csv")
candidate_movie_data = pd.read_csv("data/deployment_data/candidate_movie_data.csv")

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Define the input schema for the user to select movies
class RecommendationRequest(BaseModel):
    input_movie_ids: List[int]  # Accepts a list of up to 3 movie IDs

@app.get("/", response_class=HTMLResponse)
def get_movie_page(request: Request):
    # Fetch the list of movie titles to populate the dropdowns
    movie_titles = input_movie_dropdown['InputMovie_title'].tolist()
    return templates.TemplateResponse("index.html", {"request": request, "movie_titles": movie_titles})

@app.get("/test")
def test_route():
    return {"message": "FastAPI is working!"}


@app.get("/dropdown")
def get_input_movie_dropdown():
    try:
        # Return the dropdown data (Input Movie Titles)
        return {"movies": input_movie_dropdown.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        # Ensure the user selects 1 to 3 input movies
        if len(request.input_movie_ids) < 1 or len(request.input_movie_ids) > 3:
            raise HTTPException(status_code=400, detail="Please select 1 to 3 input movies.")

        input_movie_titles = []
        candidate_movie_scores = []

        # For each input movie, find the top candidate movie and calculate the match score
        for movie_id in request.input_movie_ids:
            # Extract the input movie data
            input_movie = candidate_movie_data[candidate_movie_data['InputMovie_id'] == movie_id]

            # Get the top candidate movie for each input movie
            top_candidate_movie = candidate_movie_data[candidate_movie_data['InputMovie_id'] == movie_id]

            # Prepare the movie and pairwise features for model prediction
            movie_features = torch.tensor([
                top_candidate_movie['normalized_CandidateMovie_vote_count'].values[0],
                top_candidate_movie['normalized_CandidateMovie_log_popularity'].values[0]
            ], dtype=torch.float32).unsqueeze(0)

            pairwise_features = torch.tensor([
                top_candidate_movie['normalized_shared_genres'].values[0],
                top_candidate_movie['normalized_shared_actors'].values[0],
                top_candidate_movie['normalized_shared_directors'].values[0]
            ], dtype=torch.float32).unsqueeze(0)

            # Make predictions using the model
            with torch.no_grad():
                score = model(movie_features, pairwise_features).item()

            # Store the results
            input_movie_titles.append(input_movie['InputMovie_title'].values[0])
            candidate_movie_scores.append((top_candidate_movie['CandidateMovie_title'].values[0], score))

        # Calculate the best candidate movie by averaging the match scores from selected input movies
        best_match = max(candidate_movie_scores, key=lambda x: x[1])

        recommendation = {
            "input_movies": input_movie_titles,
            "recommended_movie": best_match[0],
            "match_score": best_match[1]
        }

        return recommendation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))