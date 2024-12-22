#!/bin/bash
# This script will rebuild the docker image, push to GCP, and run the GCP Run

# Build the Docker image with the correct architecture
docker build --platform linux/amd64 -t gcr.io/movie-network-recommender/movie-recommendation-api:latest .

# Push the Docker image to Google Container Registry
docker push gcr.io/movie-network-recommender/movie-recommendation-api:latest

# Deploy the Docker image to Google Cloud Run
gcloud run deploy movie-recommendation-api \
  --image gcr.io/movie-network-recommender/movie-recommendation-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
  --memory 3Gi  # Adjust as needed
