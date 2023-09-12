#!/usr/bin/env python

import utility
import transformations
from google.cloud import bigquery
from datetime import datetime
import pandas as pd

# Start time for execution time calculation
startTime = datetime.now()

if __name__ == "__main__":
    
    # BigQuery client setup
    print("Retrieving GCP credentials")
    # Replace the path with your local credentials JSON file
    gcp_credentials = utility.get_gcp_credentials_local("path/to/your/credentials.json")
    bq_client = bigquery.Client(credentials=gcp_credentials, project=gcp_credentials.project_id)
    utility.calculate_time("Credentials fetched", startTime)

    # Extract raw data - user and product IDs - from BigQuery
    dataset_id = gcp_credentials.project_id + "." + "your_dataset_name"
    columns_to_pull = ['user_id', 'product_id']
    # Load raw recommender system data - user IDs and product IDs
    print("Fetching raw user and product data")
    rec_df_raw = utility.load_table_from_bigquery(bq_client, dataset_id, "stg_rec_sys_raw", columns_to_pull)
    utility.calculate_time("Raw rec sys data fetched", startTime)

    # Generating raw user-item interaction matrix
    print("Generating raw user-item matrix")
    raw_user_item_matrix = transformations.generate_user_item_matrix(rec_df_raw)
    utility.calculate_time("Raw user-item matrix generated", startTime)

    # Performing data wrangling and cleaning on the raw matrix
    print("Processing raw user-item matrix")
    processed_user_item_matrix = transformations.preprocess_user_item_matrix(raw_user_item_matrix)
    utility.calculate_time("Processing on user-item matrix complete", startTime)

    # Generating item-item similarity matrix
    print("Generating item-item similarity matrix")
    item_similarity_matrix = transformations.generate_item_similarity_matrix(processed_user_item_matrix, "jaccard")
    utility.calculate_time("Item similarity matrix generated", startTime)

    # Generating the final recommendations dataframe
    print("Generating final recommendations dataframe")
    final_recommendation_df = transformations.generate_final_recommendation_df(processed_user_item_matrix, item_similarity_matrix)
    utility.calculate_time("Final recommendations dataframe generated", startTime)

    final_recommendation_df = final_recommendation_df.convert_dtypes()
    print("Writing recommendation dataframe to the database")   

    # Pass the table name and database connection string as arguments
    utility.write_df_to_database(final_recommendation_df, "your_table_name", "your_db_connection_string")
    utility.calculate_time("Data written to database", startTime)

    # To calculate the performance of the recommendation system
    utility.get_hit_rate(final_recommendation_df, bq_client, dataset_id)
