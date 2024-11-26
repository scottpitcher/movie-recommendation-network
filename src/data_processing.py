# data_processing.py
import os
import gc
import logging
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import min, max, col
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    filename="logs/data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting data processing...")

def setup_spark():
    """Initialize Spark session."""
    logging.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("MovieRecommendation") \
        .config("spark.master", "local[*]") \
        .getOrCreate()
    logging.info("Spark session initialized.")
    return spark

def load_data():
    """Load data from CSV and prepare initial DataFrame."""
    logging.info("Loading data...")
    node_df = pd.read_csv("data/processed_data/full_data_nodes.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    spark_df = spark.createDataFrame(node_df)
    logging.info("Data loaded and converted to Spark DataFrame.")
    return spark_df

def load_shared_matrix():
    """Load shared features matrix."""
    logging.info("Loading shared matrix...")
    shared_features = np.load("data/shared_matrix.npy").astype(float)
    logging.info(f"Shared matrix of shape {shared_features.shape} loaded.")
    return shared_features

def create_pairwise_df(shared_features, movie_df):
    """Create a pairwise DataFrame from the shared features matrix."""
    logging.info("Creating pairwise DataFrame...")
    movie_ids = movie_df.select("id").orderBy("id").rdd.flatMap(lambda x: x).collect()
    assert len(movie_ids) == shared_features.shape[0], "Number of movie IDs and shared features matrix size mismatch."

    pairwise_list = [
        (
            movie_ids[i],  # InputMovie_id
            movie_ids[j],  # CandidateMovie_id
            float(shared_features[i, j, 0]),  # shared_genres
            float(shared_features[i, j, 1]),  # shared_actors
            float(shared_features[i, j, 2])   # shared_directors
        )
        for i in range(len(movie_ids))
        for j in range(len(movie_ids))
    ]

    schema = StructType([
        StructField("InputMovie_id", IntegerType(), True),
        StructField("CandidateMovie_id", IntegerType(), True),
        StructField("shared_genres", DoubleType(), True),
        StructField("shared_actors", DoubleType(), True),
        StructField("shared_directors", DoubleType(), True)
    ])

    pairwise_df = spark.createDataFrame(pairwise_list, schema=schema)
    del pairwise_list
    gc.collect()
    logging.info("Pairwise DataFrame created.")
    return pairwise_df

def join_dataframes(pairwise_df, movie_df):
    """Join pairwise DataFrame with movie-level DataFrame."""
    logging.info("Joining DataFrames...")
    # Prepare renamed DataFrames for joining
    InputMovie_df = movie_df.withColumnRenamed("id", "InputMovie_id") \
                        .withColumnRenamed("title", "InputMovie_title") \
                        .withColumnRenamed("log_popularity", "InputMovie_log_popularity") \
                        .withColumnRenamed("popularity", "InputMovie_popularity") \
                        .withColumnRenamed("vote_average", "InputMovie_vote_average") \
                        .withColumnRenamed("vote_count", "InputMovie_vote_count").repartition(200)

    CandidateMovie_df = movie_df.withColumnRenamed("id", "CandidateMovie_id") \
                        .withColumnRenamed("title", "CandidateMovie_title") \
                        .withColumnRenamed("log_popularity", "CandidateMovie_log_popularity") \
                        .withColumnRenamed("vote_average", "CandidateMovie_vote_average") \
                        .withColumnRenamed("vote_count", "CandidateMovie_vote_count") \
                        .withColumnRenamed("degree", "CandidateMovie_degree").repartition(200)

    # Join pairwise_df with InputMovie_df and CandidateMovie_df
    merged_df = pairwise_df.join(InputMovie_df, on="InputMovie_id", how="inner").repartition(200)
    merged_df = merged_df.join(CandidateMovie_df, on="CandidateMovie_id", how="inner").repartition(200)

    # Select the columns needed
    merged_df = merged_df.select(
        "InputMovie_id", "InputMovie_title", "InputMovie_popularity", "InputMovie_vote_average","InputMovie_vote_count",
        "CandidateMovie_id", "CandidateMovie_title", "CandidateMovie_popularity", "CandidateMovie_vote_average", "CandidateMovie_vote_count",
        "shared_genres", "shared_actors", "shared_directors"
    ).repartition(200)
    
    logging.info("DataFrames joined successfully.")
    return merged_df

def normalize_features(merged_df, variables_to_normalize):
    """Normalize specified variables using min-max scaling."""
    logging.info("Normalizing features...")
    for variable in variables_to_normalize:
        min_val, max_val = merged_df.select(min(col(variable)), max(col(variable))).first()
        normalized_column = f"normalized_{variable}"
        merged_df = merged_df.withColumn(
            normalized_column,
            (col(variable) - min_val) / (max_val - min_val)
        )
        logging.info(f"Normalized {variable} into {normalized_column}.")
    return merged_df

if __name__ == "__main__":
    try:
        spark = setup_spark()
        movie_df = load_data()
        shared_features = load_shared_matrix()
        pairwise_df = create_pairwise_df(shared_features, movie_df)

        # Join data and normalize
        merged_df = join_dataframes(pairwise_df, movie_df)
        variables_to_normalize = [
            "CandidateMovie_popularity",
            "CandidateMovie_vote_count",
            "shared_genres",
            "shared_actors",
            "shared_directors"
        ]
        merged_df = normalize_features(merged_df, variables_to_normalize)

        # Save or inspect the final DataFrame
        logging.info("Data processing completed.")
        merged_df.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
