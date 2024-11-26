# shared_features.py
print("Importing packages...")
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pyspark.sql import SparkSession
print("Completed!")

# Load movie IDs and initialize matrix
df = pd.read_csv("data/full_data_binned_log.csv")  # Replace with your dataset path
movie_ids = df["id"].tolist()  # Assuming `df` contains the list of all movie IDs
n = len(movie_ids)

# Initialize a 3D matrix of shape (n, n, 3)
shared_matrix = np.zeros((n, n, 3))

# Create a mapping from movie ID to index
movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Connect to Neo4j
graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))

print("Initialized 3D shared matrix of shape:", shared_matrix.shape)

# Shared Genres
print("\nCalculating shared genres...")
query_shared_genres = """
MATCH (m1:Movie)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(m2:Movie)
WHERE m1 <> m2
RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(g) AS shared_genres
"""
shared_genres = graph.run(query_shared_genres).to_data_frame()

print(f"Processing {len(shared_genres)} shared genre relationships...")
for _, row in tqdm(shared_genres.iterrows(), total=len(shared_genres), desc="Shared Genres"):
    i = movie_index[row["movie1_id"]]
    j = movie_index[row["movie2_id"]]
    shared_matrix[i, j, 0] = row["shared_genres"]  # Assign to the first layer

# Shared Actors
print("\nCalculating shared actors...")
query_shared_actors = """
MATCH (m1:Movie)-[:ACTED_IN]-(a:Actor)-[:ACTED_IN]-(m2:Movie)
WHERE m1 <> m2
RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(a) AS shared_actors
"""
shared_actors = graph.run(query_shared_actors).to_data_frame()

print(f"Processing {len(shared_actors)} shared actor relationships...")
for _, row in tqdm(shared_actors.iterrows(), total=len(shared_actors), desc="Shared Actors"):
    i = movie_index[row["movie1_id"]]
    j = movie_index[row["movie2_id"]]
    shared_matrix[i, j, 1] = row["shared_actors"]  # Assign to the second layer

# Shared Directors
print("\nCalculating shared directors...")
query_shared_directors = """
MATCH (m1:Movie)-[:DIRECTED]-(d:Director)-[:DIRECTED]-(m2:Movie)
WHERE m1 <> m2
RETURN m1.id AS movie1_id, m2.id AS movie2_id, COUNT(d) AS shared_directors
"""
shared_directors = graph.run(query_shared_directors).to_data_frame()

print(f"Processing {len(shared_directors)} shared director relationships...")
for _, row in tqdm(shared_directors.iterrows(), total=len(shared_directors), desc="Shared Directors"):
    i = movie_index[row["movie1_id"]]
    j = movie_index[row["movie2_id"]]
    shared_matrix[i, j, 2] = row["shared_directors"]  # Assign to the third layer

# Save the matrix
output_file = "data/shared_matrix.npy"
np.save(output_file, shared_matrix)
print(f"\nShared matrix of shape {shared_matrix.shape} saved to '{output_file}'")

query_node_degree = """
MATCH (m:Movie)
RETURN m.id AS id, size([(m)--() | 1]) AS degree
"""
# Execute the query and convert the results to a DataFrame
node_degree = graph.run(query_node_degree).to_data_frame()
print(node_degree.degree.value_counts().sort_index(ascending=True))

node_df = pd.merge(df, node_degree, on = "id", how = 'left')
node_df.head()

node_df.to_csv("data/full_data_nodes.csv")
