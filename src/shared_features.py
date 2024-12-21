import logging
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

log_dir = os.path.join(os.getcwd(), "logs")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "shared_features.log")),  # Save log to logs directory
        logging.StreamHandler()  # Also print logs to console
    ]
)
logging.info("Starting shared_features.py")

try:
    # Load the environment variables
    load_dotenv()
    NEO4J_URL = os.getenv("NEO4J_URL")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    logging.info("Loaded environment variables successfully")

    # Connect to Neo4j
    graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    logging.info("Connected to Neo4j successfully")

    # Load movie IDs
    df = pd.read_csv("data/processed_data/full_data_binned_log.csv")
    movie_ids = df["id"].tolist()
    n = len(movie_ids)
    logging.info(f"Loaded {n} movie IDs from dataset")

    # Initialize shared features matrix
    shared_matrix = np.zeros((n, n, 3))
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    logging.info("Initialized shared matrix with shape %s", shared_matrix.shape)

    # Calculate shared genres
    logging.info("Starting shared genres computation")
    query_shared_genres = """
    MATCH (m1:Movie)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(m2:Movie)
    WHERE m1 <> m2
    RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(g) AS shared_genres
    """
    shared_genres = graph.run(query_shared_genres).to_data_frame()
    logging.info(f"Retrieved {len(shared_genres)} shared genres relationships")

    for _, row in tqdm(shared_genres.iterrows(), total=len(shared_genres), desc="Shared Genres"):
        i = movie_index[row["movie1_id"]]
        j = movie_index[row["movie2_id"]]
        shared_matrix[i, j, 0] = row["shared_genres"]

    # Calculate shared actors
    logging.info("Starting shared actors computation")
    query_shared_actors = """
    MATCH (m1:Movie)-[:ACTED_IN]-(a:Actor)-[:ACTED_IN]-(m2:Movie)
    WHERE m1 <> m2
    RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(a) AS shared_actors
    """
    shared_actors = graph.run(query_shared_actors).to_data_frame()
    logging.info(f"Retrieved {len(shared_actors)} shared actors relationships")

    for _, row in tqdm(shared_actors.iterrows(), total=len(shared_actors), desc="Shared Actors"):
        i = movie_index[row["movie1_id"]]
        j = movie_index[row["movie2_id"]]
        shared_matrix[i, j, 1] = row["shared_actors"]

    # Calculate shared directors
    logging.info("Starting shared directors computation")
    query_shared_directors = """
    MATCH (m1:Movie)-[:DIRECTED]-(d:Director)-[:DIRECTED]-(m2:Movie)
    WHERE m1 <> m2
    RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(d) AS shared_directors
    """
    shared_directors = graph.run(query_shared_directors).to_data_frame()
    logging.info(f"Retrieved {len(shared_directors)} shared directors relationships")

    for _, row in tqdm(shared_directors.iterrows(), total=len(shared_directors), desc="Shared Directors"):
        i = movie_index[row["movie1_id"]]
        j = movie_index[row["movie2_id"]]
        shared_matrix[i, j, 2] = row["shared_directors"]

    # Save the shared matrix
    output_file = "data/processed_data/shared_matrix.npy"
    np.save(output_file, shared_matrix)
    logging.info(f"Saved shared matrix to {output_file}")

    # Calculate node degrees
    logging.info("Calculating node degrees")
    query_node_degree = """
    MATCH (m:Movie)
    RETURN m.id AS id, size([(m)--() | 1]) AS degree
    """
    node_degree = graph.run(query_node_degree).to_data_frame()
    node_df = pd.merge(df, node_degree, on="id", how="left")
    logging.info("Node degree calculation completed")

    node_df.to_csv("data/processed_data/full_data_nodes.csv", index=False)
    logging.info("Saved node data to 'data/full_data_nodes.csv'")

except Exception as e:
    logging.error("An error occurred: %s", e, exc_info=True)
finally:
    logging.info("shared_features.py script completed")
