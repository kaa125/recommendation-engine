'''
Utility functions e.g. fetching data, uploading data etc.
'''

import json
from google.oauth2 import service_account
from datetime import datetime
from sqlalchemy import create_engine
import time

def get_gcp_credentials(credentials_string):
    ''' Returns Google Cloud credentials from JSON string.

    Args:
        credentials_string: JSON string representing GCP service account JSON
    
    Returns:
        GCP Credentials object
    '''
    try:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(credentials_string))
        print("GCP credentials retrieved successfully")
    except Exception as exception_message:
        print(exception_message)
        print("could not retrieve GCP credentials")
        credentials = None
    return credentials


def load_table_from_bigquery(bq_client, dataset_id, table_name, columns, where_clause = ""):
    ''' Selects columns passed as argument from the given table name

    Args:
        bq_client: BigQuery client object
        dataset_id: ID of dataset in BigQuery
        table_name: name of table to load from BigQuery
        colums: fields to load from the table
        where_clause: optional argument for filtering data
    
    Returns:
        dataframe with loaded table data
    '''
    query = f"""
      SELECT {','.join(columns)}
      FROM `dastgyr-data-warehouse.{dataset_id}.{table_name}` """ + where_clause
    print(query)
    df = bq_client.query(query).to_dataframe()
    print("data loaded from " + table_name)
    return df


def calculate_time(message, start_time):
    '''A utility function which prints out the time taken since
    a specified startTime - for checking script's execution time
    Args:
        message: string message to print before the time
        start_time: the starting time of the main script
    Returns:
        a message with the time taken to execute till that instant/when fucntion is called
    '''
    print("=============================\n" + message + "\n"
    + str(datetime.now() - start_time) + 
    "\n=============================\n"
    )



def write_df_to_database(df_in, table_name, db_connection_string):
    '''this functions writes the input dataframe to a table in the connected database
    
    Args:
        df_in - the dataframe to be written to the table
        table_name - table name to write the data in
        db_connection_string - connection string to connect to database
    '''
    df_in['updated_at'] = datetime.now()
    df_in['is_current'] =  1

    # establish connection with db
    engine = create_engine(db_connection_string)
    conn = engine.connect()

    retries_attempted = 0
    to_insert = True # tracks whether table has yet to be inserted

    while retries_attempted < 3 and to_insert == True:
        try:
            start = time.time()
            rows_inserted = df_in.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize = 10000)
            end = time.time()

            print("write_df_to_database::Time taken for writing " + str(rows_inserted) + " records: ", end - start, "s")
            to_insert = False

        except Exception as e:
            print("write_df_to_database::exception",e)
            retries_attempted+=1

    conn.close()
