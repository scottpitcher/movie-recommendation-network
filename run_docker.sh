docker run -v $(pwd):/app -p 8080:8000 movie-recommendation-api uvicorn app:app --host 0.0.0.0 --port 8000 --reload
