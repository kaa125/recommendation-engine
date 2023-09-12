'''
Functions for pre and post processing
'''

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from tqdm.auto import tqdm

def pre_process(df, count_x):
    '''
    Implement pre-processing steps to combine and transform data in order to run 
    the FP-growth algorithm
    
    Arguments:
    - df: DataFrame containing order and product data.
    - count_x: Threshold to filter out orders with a low number of products.

    Returns:
    - DataFrame containing order IDs paired with a list of transaction items (product IDs).
    '''
    # Dropping null-valued rows
    df.dropna(axis=0, how='all', inplace=True)
    
    # Get counts of product IDs per order
    count_product_per_order = pd.DataFrame(df.groupby(['order_id'])['product_id'].count())
    
    # Filtering orders with less than count_x products
    orders_with_less_than_x_products = pd.DataFrame(count_product_per_order.reset_index()\
        [count_product_per_order.reset_index()['product_id'] < count_x]['order_id'])
    
    # Removing orders with less than X products from the main DataFrame
    df = pd.merge(df, orders_with_less_than_x_products, on=['order_id'], how="outer", indicator=True)\
        .query('_merge=="left_only"')
    
    # Cleaning up columns
    df.drop('_merge', axis=1, inplace=True)
    
    # Generate a DataFrame of order IDs against transaction items (product IDs)
    df = pd.DataFrame(df.groupby('order_id')['product_id'].apply(list))
    
    return df

def generate_items_list(df_in):
    '''
    Generate a list of transactions from a DataFrame of order IDs against product ID lists.
    
    Arguments:
    - df_in: Input DataFrame containing order IDs against product ID lists.
    
    Returns:
    - A list of lists where each list contains product IDs corresponding to a single order ID.
    '''
    transaction_list = df_in['product_id'].to_list()
    return transaction_list

def encode_transactions(transaction_list):
    '''
    One-hot encodes the list of transactions to prepare for the FP-growth algorithm.
    
    Arguments:
    - transaction_list: A list of lists where each list contains product IDs corresponding to a single order ID.
    
    Returns:
    - One-hot encoded DataFrame showing each product ID's occurrence as a Boolean value for every order ID.
    '''
    transaction_encoder = TransactionEncoder()
    transformed_array = transaction_encoder.fit(transaction_list).transform(transaction_list)
    df = pd.DataFrame(transformed_array, columns=transaction_encoder.columns_)
    return df

def generate_result_itemsets(encoded_df, min_support):
    '''
    Applies the FP-growth algorithm to the one-hot encoded input DataFrame with the specified minimum support value
    and returns frequent itemsets along with their support values.
    
    Arguments:
    - encoded_df: One-hot encoded DataFrame of transaction items.
    - min_support: Minimum support metric value to use in the FP-growth algorithm.
    
    Returns:
    - A DataFrame of frequent itemsets along with their support values.
    '''
    res = fpgrowth(encoded_df, min_support=min_support, use_colnames=True, max_len=10)
    return res

def process_results(result_set):
    '''
    Post-processing on the result set generated from the FP-growth algorithm.
    '''
    # Add a length column to track the size of the result set
    result_set['len'] = result_set['itemsets'].str.len()
    
    # Filtering results with more than 2 items
    df = result_set[result_set.len > 2]
    return df

def generate_final_dataframe(processed_results, run_ids):
    '''
    Generates the final format to write into the database with product IDs mapped to their respective recommended products.
    
    Arguments:
    - processed_results: Input DataFrame containing itemsets.
    - run_ids: List of product IDs to generate recommendations for.
    
    Returns:
    - DataFrame of final results.
    '''
    dic = {}
    for id in tqdm(run_ids):
        # Get rows where the ID exists in the itemsets and sort by length and support value (descending order)
        subset = processed_results[processed_results['itemsets'].apply(lambda x: id in x)].sort_values(by=['support','len'], ascending=False)
        
        # Get the top 3 frozensets in the subset - can vary this to get more value e.g. 4
        top_n = subset.head(3)['itemsets'].values
        
        # Combine the list of frozensets into one single set and exclude the current ID from the result
        unique_set = set().union(*top_n)
        unique_set.remove(id)  # Exclude self ID from recommendations set
        dic[id] = unique_set

    df = pd.DataFrame.from_dict(
        dic, 
        orient='index'
    )
    
    # Reset the index to turn product IDs from index to column
    df = df.reset_index().rename(columns={'index':'product_id'})
    
    # Convert columns to rows to have a denormalized-type table of product ID with recommendations in rows
    df_melted = df.melt(id_vars=['product_id'], value_vars=df.columns, value_name='recommended_product_id')
    
    # Dropping the redundant column
    df_melted.drop('variable', axis=1, inplace=True)
    
    # Arranging the table by product IDs
    df_melted.sort_values(by='product_id', inplace=True)
    
    # Dropping rows with NaN values/null recommendations
    df_melted.dropna(subset=['recommended_product_id'], inplace=True)
    
    # Casting float values to int
    df_melted = df_melted.astype({'recommended_product_id': 'int'})
    
    return df_melted