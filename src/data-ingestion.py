# data_ingestion.py
print("Importing packages...")
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import pandas as pd
import os
from tqdm import tqdm  # Progress tracking
import ast
print("Completed!")

# Load environment variables
load_dotenv()

# Neo4j Credentials
NEO4J_URL = os.getenv("NEO4J_URL")  # Neo4j connection string (e.g., bolt://localhost:7687)
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")  # Default Neo4j username
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Password for Neo4j

# Function to connect to Neo4j
def connect_to_neo4j():
    try:
        graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        print("Connected to Neo4j successfully!")
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        exit(1)

# Load preprocessed data
def load_data(file_path):
    """
    Load the preprocessed dataset for ingestion.
    """
    try:
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Dataset with {len(df)} records loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit(1)

# Clear existing Neo4j database
def clear_neo4j_database(graph):
    """
    Clear all nodes and relationships in the Neo4j database.
    """
    print("Clearing the Neo4j database...")
    graph.run("MATCH (n) DETACH DELETE n")
    print("Neo4j database cleared.")

# Ingest data into Neo4j
def ingest_data(graph, df):
    """
    Ingest movies, genres, directors, actors, and other relationships into Neo4j.
    """
    print("Ingesting data into Neo4j...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting records"):
        # Parse genres
        genres = ast.literal_eval(row["genres"]) if pd.notna(row["genres"]) else []

        # Create Movie node
        movie_node = Node(
            "Movie",
            id=row["id"],
            title=row["title"],
            release_year=int(row["release_date"][:4]) if pd.notna(row["release_date"]) else None,
            popularity=row["popularity"],
            log_popularity=row["log_popularity"],
            popularity_bin=row["popularity_bin"],
            vote_average=row["vote_average"],
            vote_count=row["vote_count"]
        )
        graph.merge(movie_node, "Movie", "id")

        # Create Genre nodes and relationships
        for genre in genres:
            genre_node = Node("Genre", name=genre)
            graph.merge(genre_node, "Genre", "name")
            rel = Relationship(movie_node, "BELONGS_TO", genre_node)
            graph.merge(rel)

        # Create Year node and relationship
        if movie_node["release_year"]:
            year_node = Node("Year", value=movie_node["release_year"])
            graph.merge(year_node, "Year", "value")
            rel = Relationship(movie_node, "RELEASED_IN", year_node)
            graph.merge(rel)

        # Create PopularityBin node and relationship
        if row["popularity_bin"]:
            pop_bin_node = Node("PopularityBin", name=row["popularity_bin"])
            graph.merge(pop_bin_node, "PopularityBin", "name")
            rel = Relationship(movie_node, "HAS_POPULARITY_BIN", pop_bin_node)
            graph.merge(rel)

        # Add Director node and relationship
        if "director" in row and pd.notna(row["director"]):
            director_node = Node("Director", name=row["director"])
            graph.merge(director_node, "Director", "name")
            rel = Relationship(director_node, "DIRECTED", movie_node)
            graph.merge(rel)

        # Add Actor nodes and relationships
        if "actors" in row and pd.notna(row["actors"]):
            actors = ast.literal_eval(row["actors"])
            for actor in actors:
                actor_node = Node("Actor", name=actor)
                graph.merge(actor_node, "Actor", "name")
                rel = Relationship(actor_node, "ACTED_IN", movie_node)
                graph.merge(rel)

    print("Data ingestion completed successfully!")

# Main function to orchestrate the ingestion
def main():
    """
    Main function to execute the data ingestion pipeline.
    """
    # Neo4j connection
    graph = connect_to_neo4j()

    # Load data
    data_file = "data/training_data/full_data_binned_log.csv"
    df = load_data(data_file)

    # Clear Neo4j database
    clear_neo4j_database(graph)

    # Ingest data
    ingest_data(graph, df)

if __name__ == "__main__":
    main()
