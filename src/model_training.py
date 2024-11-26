# model_training.py
import os
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.model_architecture import MovieRankingModel

# Set up logging
logging.basicConfig(
    filename="logs/model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting model training...")

# Define Dataset class
class MovieDataset(Dataset):
    def __init__(self, data_file):
        logging.info(f"Loading data from {data_file}...")
        self.data = pd.read_csv(data_file)
        logging.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Split features into movie-level and pairwise-level
        movie_features = torch.tensor([
            row['normalized_CandidateMovie_vote_count'],
            row['normalized_CandidateMovie_log_popularity']
        ], dtype=torch.float32)

        pairwise_features = torch.tensor([
            row['normalized_shared_genres'],
            row['normalized_shared_actors'],
            row['normalized_shared_directors']
        ], dtype=torch.float32)

        label = torch.tensor(row['match_score'], dtype=torch.float32)

        return movie_features, pairwise_features, label

def load_config():
    """Load configuration from YAML."""
    try:
        with open("config/training_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully: {config}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}", exc_info=True)
        raise e

def train_model(model, train_loader, criterion, optimizer, config):
    """Train the model."""
    model.train()
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for movie_features, pairwise_features, labels in train_loader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                movie_features, pairwise_features, labels = (
                    movie_features.cuda(),
                    pairwise_features.cuda(),
                    labels.cuda()
                )

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(movie_features, pairwise_features)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def save_model(model, model_path):
    """Save the trained model."""
    try:
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

def main():
    load_dotenv()

    # Load configuration
    config = load_config()

    # Load training data
    training_data_path = config["training_data_path"]
    dataset = MovieDataset(training_data_path)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model, loss, and optimizer
    movie_feature_dim = 2  # Features for the movie branch
    pairwise_feature_dim = 3  # Features for the pairwise branch
    model = MovieRankingModel(movie_feature_dim, pairwise_feature_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    logging.info("Model initialized.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    logging.info("Optimizer and loss function initialized.")

    # Train the model
    train_model(model, train_loader, criterion, optimizer, config)

    # Save the trained model
    model_path = config["model_save_path"]
    save_model(model, model_path)
    logging.info("Training completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
