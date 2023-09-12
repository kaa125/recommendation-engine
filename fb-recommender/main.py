#!/usr/bin/env python

## imports
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

import utility
import processing
from google.cloud import bigquery
from datetime import datetime, timedelta
from tqdm import tqdm
from pytz import timezone    

# environment
from dotenv import load_dotenv
import os

# loading environment variables and setting constants
load_dotenv()
GCP_CREDS_STRING = os.environ.get('GCP_CREDS_STRING')
DATASET_ID = os.environ.get('DATASET_ID')
# table to write final dataframe into in the db
TABLE_NAME = os.environ.get('TABLE_NAME')
DB_CONNECTION_STRING = os.environ.get('DB_CONNECTION_STRING')
# for execution time calculation
START_TIME = datetime.now()
# time range for pulling records (in days) - order_items, orders etc.
# setting to 3 months ~ 91 days
TIME_RANGE = 91
TODAY = datetime.today()
# string representing extraction start date in Y-m-d format
EXTRACTION_START_DATE = (TODAY - timedelta(days=TIME_RANGE)).strftime("%Y-%m-%d")

if __name__ == "__main__":
    # fetch credentials
    gcp_credentials = utility.get_gcp_credentials(GCP_CREDS_STRING)
    utility.calculate_time("Credentials fetched", START_TIME)

    # BigQuery client setup
    bq_client = bigquery.Client(credentials=gcp_credentials, project=gcp_credentials.project_id)
    utility.calculate_time("BigQuery client set up", START_TIME)

    # Get order items data
    print("Fetching input data...")
    df_in = utility.load_table_from_bigquery(
        bq_client=bq_client, 
        dataset_id=DATASET_ID, 
        table_name='stg_rec_sys_raw',
        columns=['order_id','product_id'], 
        where_clause=f"WHERE created_at > '{EXTRACTION_START_DATE}' LIMIT 800000"
    )
    utility.calculate_time("Raw data fetched", START_TIME)

    print("Pre-processing dataframe...")
    processed_df = processing.pre_process(df_in, count_x=4)
    utility.calculate_time("Pre-processing done", START_TIME)

    # Generate list of transaction items
    print("Generating transaction list...")
    transaction_list = processing.generate_items_list(processed_df)
    utility.calculate_time("Transaction list generated", START_TIME)

    # Generate one-hot encoded dataframe
    print("One-hot encoding transactions...")
    encoded_df = processing.encode_transactions(transaction_list)
    utility.calculate_time("Transactions encoded", START_TIME)

    # Generate result set
    # Setting minimum support threshold for FP-Growth algorithm
    min_support = 0.0001
    print("Generating result set...")
    result_set = processing.generate_result_itemsets(encoded_df, min_support=min_support)
    utility.calculate_time("Result set generated", START_TIME)

    # Post-processing result set
    print("Post-processing result set...")
    processed_results = processing.process_results(result_set)
    utility.calculate_time("Result set post processed", START_TIME)

    # Get unique product IDs from the itemsets only
    list_of_itemsets = processed_results.itemsets.to_list()
    product_id_set = set().union(*list_of_itemsets)
    run_ids = list(product_id_set)
    print(len(product_id_set))
    print(len(run_ids))
    
    print("Generating dictionary for final dataframe")
    final_df = processing.generate_final_dataframe(processed_results, run_ids)
    utility.calculate_time("Final dataframe generated", START_TIME)

    # Write dataframe to the database
    print("Writing dataframe to database")
    utility.write_df_to_database(final_df, 
            table_name=TABLE_NAME, 
            db_connection_string=DB_CONNECTION_STRING
    )
    utility.calculate_time("Data written to database", START_TIME)
