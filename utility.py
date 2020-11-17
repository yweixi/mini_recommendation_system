#!/usr/bin/env/ python
import os
import time
import sklearn
import numpy as np
import pandas as pd
import surprise
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt
from scipy import stats
from surprise import NMF
from surprise import Reader
from surprise import Dataset
from surprise import KNNWithMeans
from collections import defaultdict
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# this method returns a subset of the input file
def sample_dateset_by_percentile(input_df, target_user_size, target_item_size):

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

def get_top_n(predictions, n=5):
    
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n