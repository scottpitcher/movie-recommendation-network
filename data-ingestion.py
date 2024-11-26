        # For safer evaluation of stringified lists
print("Importing packages...")
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Progress tracking
import ast     
print("Completed!")

# Load the dataset with log-transformed popularity bins
df = pd.read_csv("data/full_data_binned_log.csv")

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")  # Neo4j connection string (e.g., bolt://localhost:7687)
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")  # Neo4j username
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Neo4j password

print("Neo4j credentials retrieved")

try:
    graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print("Connected to Neo4j successfully!")
except Exception as e:
    print("Failed to connect:", e)

# Connect to Neo4j
graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Clear existing data in Neo4j
graph.run("MATCH (n) DETACH DELETE n")

# Create nodes and relationships with progress tracking
for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting data into Neo4j"):

    # Parse genres safely
    genres = ast.literal_eval(row["genres"]) if pd.notna(row["genres"]) else []
    
    # Create movie node 
    movie_node = Node(
        "Movie",
        id=row["id"],
        title=row["title"],
        release_year=int(row["release_date"][:4]) if pd.notna(row["release_date"]) else None,
        popularity=row["popularity"],
        popularity_bin=row["popularity_bin"],
        vote_average=row["vote_average"],
        vote_count=row["vote_count"]
    )
    graph.merge(movie_node, "Movie", "id")

    # Create genre nodes and relationships
    # For loop due to possibility of multiple genre vals
    for genre in genres:
        genre_node = Node("Genre", name=genre)
        graph.merge(genre_node, "Genre", "name")
        rel = Relationship(movie_node, "BELONGS_TO", genre_node)
        graph.merge(rel)

    # Create year node and relationship
    if movie_node["release_year"]:
        year_node = Node("Year", value=movie_node["release_year"])
        graph.merge(year_node, "Year", "value")
        rel = Relationship(movie_node, "RELEASED_IN", year_node)
        graph.merge(rel)

    # Create a popularity bin node and relationship
    if row["popularity_bin"]:
        pop_bin_node = Node("PopularityBin", name=row["popularity_bin"])
        graph.merge(pop_bin_node, "PopularityBin", "name")
        rel = Relationship(movie_node, "HAS_POPULARITY_BIN", pop_bin_node)
        graph.merge(rel)

    # Add directors (if available)
    if "director" in row and pd.notna(row["director"]):
        director_node = Node("Director", name=row["director"])
        graph.merge(director_node, "Director", "name")
        rel = Relationship(director_node, "DIRECTED", movie_node)
        graph.merge(rel)

    # Add actors (if available and stored as a list in the dataset)
    if "actors" in row and pd.notna(row["actors"]):
        actors = ast.literal_eval(row["actors"])
        for actor in actors:
            actor_node = Node("Actor", name=actor)
            graph.merge(actor_node, "Actor", "name")
            rel = Relationship(actor_node, "ACTED_IN", movie_node)
            graph.merge(rel)

print("Data ingestion completed with extended entities and relationships.")
