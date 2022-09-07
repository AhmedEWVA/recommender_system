import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import requests
import json
import time
from tqdm.auto import tqdm
import math
import os, os.path

import ast
import re


import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors
import seaborn as sns

from collections import Counter


from sklearn.neighbors import NearestNeighbors

import os
import urllib.request
from numpy import dot
from numpy.linalg import norm
import argparse



def compute_distance(company_id, candidates_id):
    return

def get_rank_wrt_distance(candidate_neigh, distances):
    # rank companies by their distance, the closer the company the higher its rank
    neighbours1 = []
    neighbours2 = []
    neighbours3 = []
    for i in candidate_neigh:
        if distances[i] == 1:
            neighbours1.append(i)
        elif distances[i] == 2:
            neighbours2.append(i)
        elif distances[i] == 3:
            neighbours3.append(i)
    return neighbours1, neighbours2, neighbours3

def rank_by_location(location, candidates_ids):
    # rank companies by their location, the closer the company the higher its rank
    distances = compute_distance(location, candidates_ids)
    n1,n2,n3 = get_rank_wrt_distance(candidates_ids, distances)
    return n1+n2+n3




def get_closest( embedding, embeddings, num_neighbours):
    # based on embedding vectors compute the nearest companies
    neigh = NearestNeighbors(n_neighbors=2, radius=0.7)
    neigh.fit(embeddings)
    neighbours = neigh.kneighbors([embedding], num_neighbours, return_distance=False)
    return neighbours

def recommendation1(id, embedding, size, ids, embeddings, sizes, num_neighbours):
    # select companies with similar size then compute the most similar ones semantically
    #size, location, companies
    idx = np.where(ids==id)[0][0]
    #location = row["location"]
    if (size == 2):
        embeddings = embeddings[np.where(sizes==2)[0]]
        neighbours = get_closest( embedding, embeddings, num_neighbours)
        return neighbours
    elif (size == 1):
        #ocation = continent
        embeddings = embeddings[np.where(sizes==1)[0]]
        neighbours = get_closest(embedding, embeddings, num_neighbours)
        #neighbours = rank_by_location(company_id, neighbours, num_neighbours)
        return neighbours
    elif (size == 0):
        embeddings = embeddings[np.where(sizes==0)[0]]
        neighbours = get_closest(embedding, embeddings, num_neighbours)
        #neighbours = rank_by_location(company_id, neighbours, num_neighbours)
        return neighbours

def recommendation2(id, embedding, size, ids, embeddings, sizes, num_neighbours):
    # First compute the most similar companies based on semantic information then rank the obtained companies based size (and further companies)
    idx = np.where(ids==id)[0][0]
    neighbours = get_closest( embedding, embeddings, num_neighbours)
    #print(sizes[neighbours])
    neighbours = get_rank_wrt_size(neighbours, sizes[neighbours], size)
    return neighbours


def distance_with_size(size1, size2):
    # compute a size-relative distance between two companies, the more similar the companies are the smaller the distance
    return np.abs(size1-size2) 


def get_rank_wrt_size(candidate_neigh, sizes, size):
    # rank recommanded companies by size
    distances = [distance_with_size(size,j) for j in sizes][0]
    neighbours1 = []
    neighbours2 = []
    neighbours3 = []
    for i, ele in enumerate(candidate_neigh):
        if distances[i] == 0:
            neighbours1.append(ele)
        elif distances[i] == 1:
            neighbours2.append(ele)
        elif distances[i] == 2:
            neighbours3.append(ele)
    return neighbours1+neighbours2+neighbours3

# rank recommendations by location
"""def rank_by_location(location, candidates_ids):
    distances = compute_distance(location, candidates_ids)
    n1,n2,n3 = get_rank_wrt_distance(candidates_ids, distances)
    return n1+n2+n3"""

def to_ids(series, df):
    # this functions maps the index to the id
    series_ = series.copy(deep=True)
    for i in tqdm(range(len(series))):
        for j, ele in enumerate(series[i][0]):
            series_[i][0][j] = df.iloc[ele,list(df.columns).index("id")]
    return series_

"""def to_ids(series, df):
    series_ = series.copy()
    new_series = []
    for i,arr in enumerate(series):
        l = []
        for j, ele in enumerate(series[i][0]):
            #series_[i][0][j] = df.iloc[ele,list(df.columns).index("id")]
            l.append(df.iloc[ele,list(df.columns).index("id")])
        new_series.append(np.array(l))
    return series_"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--df_name', type=str, default='companies_embeddings.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=100, metavar='N',
                        help='The end integer')
    args = parser.parse_args()
    tqdm.pandas()  
    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')

    test_df = companies_df["id"].iloc[args.beginning: args.ending, ].copy()


    
    
    # processing embeddings (from string to floats)
    embeddings = companies_df["embeddings"].values
    embeddings_list = []
    for e in embeddings:
        embeddings_list.append(ast.literal_eval(e))
    embeddings = np.array(embeddings_list)

    sizes = companies_df["sizes"].values
    ids = companies_df["id"].values

    #locations = companies_df["locations"].values

    # compute recommendations for each company
    series_tmp = [recommendation2(ids[i], embeddings[i], sizes[i], ids[:], embeddings[:], sizes[:], 100) for i in tqdm(range(len(sizes)))]
    # map from indeces to ids
    series_ids = to_ids(series_tmp, companies_df)

    # mapping from one company's id to the ids of similar companies
    recommendations_df = pd.concat([companies_df["id"],pd.Series(series_ids)],axis=1)
    recommendations_df.columns = ["id", "recommendations"]

    #export the results into a csv
    recommendations_df.to_csv(f'recommendations.csv')