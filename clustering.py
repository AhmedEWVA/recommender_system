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
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim
import gensim.downloader

import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors
import seaborn as sns

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

import os
import urllib.request
from scipy import spatial
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
import argparse


def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)

def give_examples_of_clusters(num, clustering):
    # from each cluster select num elements randomly
    examples = []
    labels = np.unique(clustering.labels_)
    for i in labels:
        examples.append(np.random.choice(find_indices(clustering.labels_,i), size=num))
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--df_name', type=str, default='companies_embeddings.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=-1, metavar='N',
                        help='The end integer')
    parser.add_argument('--num_clusters', type=int, default=30, metavar='N',
                        help='number of clusters')
    parser.add_argument('--method', type=str, default="kmeans", metavar='N',
                        help='Clustering method')
    parser.add_argument('--column', type=str, default="embeddings", metavar='N',
                        help='Specify the column on which the clustering is applied')
    args = parser.parse_args()

    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')
    tqdm.pandas()

    # processing embeddings (from string to floats)
    embeddings = companies_df[args.column]
    embeddings_list = []
    for e in embeddings:
        embeddings_list.append(ast.literal_eval(e))
    vectors =  np.array(embeddings_list)
    
    #apply the clustering methods
    if (args.method == "kmeans"):
        clustering = KMeans(n_clusters=args.num_clusters, random_state=0).fit(vectors)
    elif (args.method == "ward"):
        connectivity = kneighbors_graph(vectors, n_neighbors=10, include_self=False)
        clustering = AgglomerativeClustering(n_clusters=args.num_clusters, connectivity=connectivity, linkage="ward").fit(vectors)


    """examples = give_examples_of_clusters(20, clustering)

    for i in examples[1]:
        print(companies_df["tokens_keywords"].values[i])
    """

    companies_df["clustering_labels"] = clustering.labels_
    #companies_df["kw_embed"] = companies_df["tokens_keywords"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 ))  

    #exporting results to csv file
    companies_df.to_csv(f'companies_clustering.csv')