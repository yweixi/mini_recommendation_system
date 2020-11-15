#!/usr/bin/env/ python
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# this method returns a subset of the input file
def sample_dateset_by_percentile(input_df, 
                                 target_user_size, 
                                 target_item_size):
    
    '''
    input_df: the dataframe should have three columns
        1. userId
        2. movieId
        3. rating
    
    target_item_size: the number of movie items we want to keep
    target_user_size: the number of users we want to keep
    '''
    # if the dataset doesn't have enough movie items to select
    if input_df['movieId'].nunique() <= target_item_size:
        print("dataset doesn't have enough items to select")
        return input_df
    
    # count the number of ratings each user makes
    user_activeness = input_df.groupby(by=['userId']).count().drop(['rating'], axis=1).rename(columns={'movieId':'# movie rated'})
    user_activeness = user_activeness.reset_index()
    
    # select user by percentile
    per_50 = user_activeness['# movie rated'].describe()['50%']
    top_popularity = user_activeness[user_activeness['# movie rated']>=per_50].drop(['# movie rated'], axis=1)
    top_popularity_selected = top_popularity.sample(n=target_user_size, random_state=1)
    input_df = pd.merge(input_df,
                        top_popularity_selected,
                        on='userId',
                        how='inner')
    
    
    # count the number of ratings each movie has
    movie_popularity = input_df.groupby(by=['movieId']).count().drop(['rating'], axis=1).rename(columns={'userId':'# user rated'})
    movie_popularity = movie_popularity.reset_index()
        
    # select movie items by percentile
    per_50 = movie_popularity['# user rated'].describe()['50%']
    top_popularity = movie_popularity[movie_popularity['# user rated']>=per_50].drop(['# user rated'], axis=1)    
    top_popularity_selected = top_popularity.sample(n=target_item_size, random_state=1)
    
    # subset the dataset and only keep movie items
    ratings_sub = pd.merge(input_df, 
                           top_popularity_selected, 
                           on='movieId', 
                           how='inner') 
    return ratings_sub

# xxxxx
def sim_matrix(ratings_df):
    item_sim_matrix = pd.DataFrame(data=cosine_similarity(ratings_df),
                                   index=ratings_df.index,
                                   columns=ratings_df.index)
    return item_sim_matrix

# xxxxx
def single_item_rating(user_id, 
                       item_id, 
                       n_neighbor, 
                       similarity_matrix, 
                       mean_ratings_df, 
                       ratings_df):
    
    # target item mean
    item_mean = np.float(mean_ratings_df.loc[mean_ratings_df.index==item_id, 'mean'])
    # target item similarity

    item_neighbors = similarity_matrix.loc[:, similarity_matrix.columns==item_id]
    # get the neightbors
   
    item_neighbors_sort = item_neighbors.sort_values(by=item_id, ascending=False).head(n_neighbor+1)
    # get the neighbor list
    
    neighbor_list = [item_neighbors_sort.index.values[i+1, ] for i in range(n_neighbor)]
    # get the neighbors' mean
    neighbor_mean = [np.float(mean_ratings_df.loc[mean_ratings_df.index==item_id, 'mean']) for item_id in neighbor_list]
    # get the neighbors' similarity
    similarity_list = [item_neighbors_sort.loc[item_neighbors_sort.index==neighbor_list[i], :].values[0][0] for i in range(n_neighbor)]
    # the the rating history of the user
    user_rating_history = ratings_df[ratings_df['userId']==user_id]
    
    # make prediction
    numerator = 0
    denominator = sum([abs(similarity) for similarity in similarity_list])
    
    if denominator == 0:
        return item_mean
    
    for i in range(len(neighbor_list)):
        similarity = similarity_list[i]
        if neighbor_list[i] in user_rating_history['movieId'].values:
            res = user_rating_history[user_rating_history['movieId']==neighbor_list[i]]['rating'].values[0]
        else:
            res = 0
        numerator += similarity*res
    
    return item_mean + numerator / denominator

# xxxxx
def user_item_rating(user_id, 
                     ratings_df, 
                     similarity_matrix, 
                     mean_ratings_df, 
                     n_neighbor=20):
    
    item_list = [item for item in similarity_matrix.index.values]
    
    item_to_rate = item_list
    
    rating_pred = {}
    for item in item_to_rate:
        
        rating_pred[item] = single_item_rating(user_id=user_id, 
                                               item_id=item, 
                                               n_neighbor=n_neighbor, 
                                               similarity_matrix=similarity_matrix, 
                                               mean_ratings_df=mean_ratings_df, 
                                               ratings_df=ratings_df)
    return rating_pred 

# xxxxx
def topn_recommendation(user_id,
                        ratings_df,
                        ratings_prediction_dict, 
                        n_recommendations):
    recommendation = []
    rating_history = ratings_df.loc[ratings_df['userId']==user_id]['movieId'].values
    for key, value in sorted(ratings_prediction_dict.items(), key=lambda kv: kv[1], reverse=True):
        if key not in rating_history:
            recommendation.append(key)
    return recommendation[:n_recommendations]

# xxxxx
def matrix_fill(ratings_df, 
                ratings_char, 
                mean_ratings_df, 
                similarity_matrix):

    ratings_pred = pd.DataFrame()
    ratings_test = ratings_char
    
    for user in ratings_test.columns:
        
        result = user_item_rating(user_id=user,
                                  ratings_df=ratings_df,
                                  similarity_matrix=similarity_matrix, 
                                  mean_ratings_df=mean_ratings_df)
        user_ratings = pd.DataFrame.from_dict(result, orient='index', columns=[user])
        
        ratings_pred = pd.concat([ratings_pred, user_ratings], axis=1)
    
    return ratings_pred.fillna(0)