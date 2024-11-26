# Neo4j Movie Recommender GNN Model 
Personal project more for learning than portfolio

## Project Overview 

#### Stage 1: Data Ingestion ✅
- Fetch movie data from TMDb through API.
- Map genres onto their respective IDs.
- Save full dataset.

#### Stage 2: Knowledge Graph Construction ✅
- Ensure Neo4j is running locally via Docker.
- Create nodes for entities (Movies, Genres), and relationships (Movies → Genres, Movies → Release Year, Movies → Popularity Scores).
- Utilize Cypher queries to ensure proper ingestion of relational data.

#### Stage 3: Feature Engineering ✅
- Utilize Cypher queries to extract features:
  - **Shared Genres:** Number of genres shared between input movies and candidate movies.
  - **Shared Actors/Directors:** Connections between movies via shared cast/crew.
  - **Graph Metrics:**
    - **Node Degree:** Number of direct relationships a movie has.
    - **PageRank:** Importance of movies in the graph structure.
- Export graph-based features from Neo4j to a pandas DataFrame.
- Label movies with a match score (0-1) for pairwise combinations.

#### Stage 4: GNN-Based Recommendation Model Development (Current Stage)
- **Graph Construction for GNN:**
  - Export nodes, edges, and adjacency matrices from Neo4j to build a graph for GNN training.
  - Use graph-based features as node attributes (e.g., popularity_bin, PageRank).
- **Model Selection:**
  - Build a GNN architecture using frameworks like PyTorch Geometric or DGL:
    - **Graph Convolutional Network (GCN):** For node embeddings.
    - **Graph Attention Network (GAT):** For weighing relationships between nodes.
  - Train the model for:
    - **Node Classification:** Classify candidate movies as recommended or not.
    - **Link Prediction (Optional):** Predict links between user-selected movies and candidates.
- **Model Evaluation:**
  - Use metrics like accuracy, F1-score (for classification), or AUC (for link prediction).

#### Stage 5: Recommendation System Deployment
- **User Input:**
  - Allow users to select 3 movies as input.
- **Real-Time Feature Extraction:**
  - Query Neo4j for candidate movies and their graph-based features dynamically.
- **Real-Time Prediction:**
  - Use the trained GNN model to recommend movies based on input.
- **Deployment:**
  - Deploy the pipeline with Docker and expose as an API using Flask or FastAPI.
 

## Data Source(s)
The Movie Database: https://www.themoviedb.org/

