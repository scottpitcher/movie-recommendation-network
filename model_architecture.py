# model_architecture.py
# This script contains the architecture for the recommender model
# While this won't be directly run, it will be used in other training/evaluation scripts

import torch
import torch.nn as nn

class MovieRankingModel(nn.Module):
    def __init__(self, movie_feature_dim, pairwise_feature_dim):
        super(MovieRankingModel, self).__init__()
        
        # Movie-level feature branch
        self.movie_branch = nn.Sequential(
            nn.Linear(movie_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Pairwise feature branch
        self.pairwise_branch = nn.Sequential(
            nn.Linear(pairwise_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combined dense layers
        self.combined = nn.Sequential(
            nn.Linear(64 + 64, 64),  # Combining outputs from both branches
            nn.ReLU(),
            nn.Linear(64, 1)        # Output relevance score
        )
    
    def forward(self, movie_features, pairwise_features):
        # Pass through branches
        movie_out = self.movie_branch(movie_features)
        pairwise_out = self.pairwise_branch(pairwise_features)
        
        # Concatenate outputs
        combined = torch.cat((movie_out, pairwise_out), dim=1)
        
        # Final prediction
        output = self.combined(combined)
        return output