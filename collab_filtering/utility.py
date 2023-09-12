import json
import time
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

def get_gcp_credentials_local(creds_fp):
    ''' Returns Google Cloud credentials from a file path.

    Args:
        creds_fp: File path to GCP service account JSON file
    
    Returns:
        GCP Credentials object
    '''
    try:
        credentials = service_account.Credentials.from_service_account_file(creds_fp)
        print("GCP credentials retrieved successfully")
    except:
        print("Could not retrieve GCP credentials")
        credentials = None
    return credentials

def load_table_from_bigquery(bq_client, dataset_id, table_name, columns, where_clause = ""):
    ''' Selects columns passed as arguments from the given table name.

    Args:
        bq_client: BigQuery client object
        dataset_id: ID of dataset in BigQuery
        table_name: Name of table to load from BigQuery
        columns: Fields to load from the table
        where_clause: Optional argument for filtering data
    
    Returns:
        Dataframe with loaded table data
    '''
    query = f"""
        SELECT {','.join(columns)}
        FROM `{dataset_id}.{table_name}` """ + where_clause
    df = bq_client.query(query).to_dataframe()
    print("Data loaded from " + table_name)
    return df

def calculate_time(message, startTime):
    '''A utility function which prints out the time taken since
    a specified startTime - for checking script's execution time.
    
    Args:
        message: String message to print before the time
        startTime: The starting time of the main script
    
    Returns:
        A message with the time taken to execute till that instant/when the function is called
    '''
    print("=============================\n" + message + "\n"
    + str(datetime.now() - startTime) + "\n=============================\n"
    )

def write_df_to_database(recommendations_df):
    '''This function writes the input dataframe to a table in the connected database.
    
    Args:
        recommendations_df: The dataframe to be written to the table
    '''
    recommendations_df['updated_at'] = datetime.now()
    recommendations_df['is_current'] = True
    # Name of table where dataframe is to be written
    TABLE_NAME = "recommendations"

    # Fetch connection string to connect to the database
    with open('./creds/database_configuration.json') as f:
        config = json.load(f)

    # Establish connection with the database
    engine = create_engine(config["connection_string"])
    conn = engine.connect()

    retries_attempted = 0
    to_insert = True # Tracks whether table has yet to be inserted

    while retries_attempted < 3 and to_insert == True:
        try:
            start = time.time()
            rows_inserted = recommendations_df.to_sql(TABLE_NAME, con=engine, index=False, if_exists='replace', chunksize = 10000)
            end = time.time()

            print("write_df_to_database::Time taken for writing " + str(rows_inserted) + " records: ", end - start, "s")
            to_insert = False

        except Exception as e:
            print("write_df_to_database::exception", e)
            retries_attempted += 1

    conn.close()

def get_hit_rate(recommendations_df, bq_client, dataset_id):
    '''This function returns the hit rate %, i.e., the number of recommendations we made and how many people 
    bought the recommended items.
    
    Args:
        recommendations_df: The dataframe containing recommendations
        bq_client: BigQuery client object
        dataset_id: ID of dataset in BigQuery
    '''
    recommendation_users = recommendations_df['user_id'].nunique()
    print("# of users recommendations are generated for: ", recommendation_users)

    # Get orders of users who are both in training and test dataset
    filter = "WHERE OrderCreatedAt >= '2022-10-10' and UserID in {}".format(tuple(recommendations_df['user_id'].unique()))
    orders_test_dataset = load_table_from_bigquery(bq_client, dataset_id, "stg_orders_denorm", ["UserID", "PRODUCT_ID"], filter)
    overlapping_users = orders_test_dataset["UserID"].nunique()
    print("# of users present in both training and test datasets: ", overlapping_users)
    print("Percentage of overlapping users: {}%".format(overlapping_users / recommendation_users * 100))

    merged_df = pd.merge(orders_test_dataset, recommendations_df, how="inner", left_on=["UserID", "PRODUCT_ID"], right_on=["user_id", "product_id"])

    hit_rate = len(merged_df.index) / len(orders_test_dataset.index) * 100
    print("Hit rate: {}%".format(hit_rate))
