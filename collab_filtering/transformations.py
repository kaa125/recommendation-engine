import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from datetime import datetime

def generate_user_item_matrix(df):
    '''
    Applies aggregations and pivoting to generate a user-item matrix
    using raw table data.
    
    Args:
    - df: raw dataframe to transform into a user-item matrix
    
    Returns:
    - User-item matrix
    '''
    # Adding product count column
    df['product_count'] = df.groupby('product_id')['product_id'].transform('count')

    # Grouping count by product id per user - this gives us the count of orders of a product from a particular user
    df = df.groupby(['user_id','product_id']).count().reset_index()

    matrix = pd.pivot_table(df, values='product_count', index='product_id', columns='user_id')

    print("Raw user-item matrix generated")
    return matrix

def preprocess_user_item_matrix(matrix):
    '''
    Pre-processing steps to transform the user-item matrix, including 
    removal of items with low interactions, data normalization, imputing null values, etc.

    Args:
    - matrix: raw user-item matrix
    
    Returns:
    - Processed and normalized user-item matrix
    '''
    # Dropping items which have only 2 interactions/views/purchases and that too with a value of one
    li_prod = []
    for product_id in matrix.index:
        # If there are at most 2 items in a row with a value of 1, drop that row/item
        if (matrix.loc[product_id] <= 1).sum() <= 2:
            li_prod.append(product_id)
    print("Dropping", len(li_prod), "items")
    matrix.drop(li_prod, axis=0, inplace=True)

    # Dropping users which have no interactions/views/purchases (formed as a result of removing items in the above process)
    li_users = []
    for user_id in matrix.columns:
        # Check whether the number of null values for each column matches the number of items - meaning no interactions
        if np.any(matrix.loc[:, user_id].isnull().sum() == len(matrix.index)):
            li_users.append(user_id)
    print("Dropping", len(li_users), "users")
    matrix.drop(li_users, axis=1, inplace=True)

    # Normalizing the matrix to be between 0 and 1 using min-max normalization
    # matrix = matrix.subtract(matrix.mean(axis=1), axis=0)

    # Imputing NaN values to zero
    matrix = matrix.fillna(0)

    print("Pre-processing on the raw user-item matrix complete")
    return matrix

def generate_item_similarity_matrix(normalized_matrix, type="cosine"):
    '''
    Calculating item-item similarity using cosine/jaccard similarity
    and generating the subsequent item-item similarity matrix.
    '''
    if type == "cosine":
        # Calculate cosine similarities
        item_similarity_arr = cosine_similarity(normalized_matrix)
    elif type == "jaccard":
        # Calculate jaccard similarities
        item_similarity_arr = 1 - pairwise_distances(normalized_matrix.to_numpy(), metric="jaccard")

    # Turn item_similarity numpy array into a pandas pivot table dataframe
    item_similarity_matrix = pd.DataFrame(item_similarity_arr, index=normalized_matrix.index, columns=normalized_matrix.index)
    return item_similarity_matrix

def get_item_recommendations(picked_userid, number_of_recommendations, user_item_matrix, item_similarity_matrix):
    '''
    Generates item recommendations for a picked user id based on the number of recommendations required.
    
    Args:
    - picked_user_id: User ID for which recommendations are to be generated
    - number_of_recommendations: Total number of recommendations to generate and use for past interaction history
    - user_item_matrix: The normalized and processed user-item matrix which associates users with items based on interactions
    - item_similarity_matrix: The item-item similarity matrix used to extract similarity scores for products
    
    Returns:
    - Recommendations for the picked user ID as specified by the number_of_recommendations
    '''
    # Series of all products, viewed or unviewed
    all_products = user_item_matrix[picked_userid]
    
    # Products that the target user has viewed
    viewed_products = all_products[all_products != 0]
    # Getting the top viewed item for the users
    viewed_products = pd.DataFrame(viewed_products.sort_values(ascending=False)).reset_index().rename(columns={picked_userid:'rating'})[:number_of_recommendations]
    
    # Recommended items dict for the picked user_id
    rec_items = {}

    for product in viewed_products['product_id']:
        # Get id and similarity value of nearest/most similar item to the current product
        series = item_similarity_matrix.loc[product] # All item similarity scores corresponding to the current product
        series = series[series.index != product] # Excluding the current product from the list for comparison
        rec_product_id = series.idxmax() # Getting the product id with the max similarity with the current product
        rec_product_similarity = item_similarity_matrix.loc[product, rec_product_id] # Getting the similarity value for the rec_prod_id
        # Adding the item to the dictionary
        rec_items[rec_product_id] = round(rec_product_similarity, 5)
    
    return_dict = {
        'user_id': picked_userid,
        'type': 1,
        'product_id': list(rec_items.keys())
    }
    return return_dict

def generate_final_recommendation_df(user_item_matrix, item_similarity_matrix):
    '''
    Generates the final data frame containing user ids, recommendation model version, and
    recommended items for each user. This data frame will be fed to a lookup table in production.
    
    Args:
    - user_item_matrix: Used to derive the number of user_ids and for feeding to the get_item_recommendations function
    - item_similarity_matrix: Used to feed the get_item_recommendations function
    
    Returns:
    - A dataframe consisting of user id, recommendation type, and recommended items for each user
    '''

    user_ids = user_item_matrix.columns
    # List of data frames to concatenate
    li_dfs = []
    # Setting the default number of recommendations to generate
    NUM_RECOMMENDATIONS = 6

    # Iterating over user id's to get recommendations for each user
    for id in user_ids:
        try:
            recommendation_dict = get_item_recommendations(picked_userid=id, 
                                    number_of_recommendations=NUM_RECOMMENDATIONS,
                                    user_item_matrix=user_item_matrix, 
                                    item_similarity_matrix=item_similarity_matrix)
            interim_df = pd.json_normalize(recommendation_dict).explode('product_id')
            li_dfs.append(interim_df)
        except Exception as e:
            print(str(id) + ":\n")
            print(e)
            break
    # Generate the final dataframe
    final_df = pd.concat(li_dfs, ignore_index=True)

    # Changing dtypes of the final dataframe
    final_df = final_df.convert_dtypes()

    return final_df
