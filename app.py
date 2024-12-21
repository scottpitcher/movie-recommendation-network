# app.py
from src.model_architecture import MovieRankingModel  # Import the model architecture
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.model_architecture import MovieRankingModel  # Import the model

# Initialize FastAPI
app = FastAPI()

# Load the model
model_path = "models/movie_ranking_model.pt"
movie_feature_dim = 2     # Number of movie-level features
pairwise_feature_dim = 3  # Number of pairwise features

# Initialize the model
model = MovieRankingModel(movie_feature_dim, pairwise_feature_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the input schema
class RecommendationRequest(BaseModel):
    movie_features: List[float] = [0.0, 0.0]
    pairwise_features: List[float] = [0.0, 0.0, 0.0] 


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        # Prepare inputs
        movie_features = torch.tensor(request.movie_features, dtype=torch.float32).unsqueeze(0)
        pairwise_features = torch.tensor(request.pairwise_features, dtype=torch.float32).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            score = model(movie_features, pairwise_features).item()

        return {"match_score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
