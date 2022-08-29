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
from gensim.models import Word2Vec
import gensim
import gensim.downloader

import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors

from collections import Counter


import os
import urllib.request
from scipy import spatial
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
import argparse


def get_description_embedding_avg(description, emmbed_dict, embed_dim):
    # compute an average over the embeddings of the words in a list
    n = len(description)
    embedding = np.zeros(embed_dim)
    for i, ele in enumerate(description):
        try:
            embedding += emmbed_dict[ele]
        except:
            pass
    return embedding/n

def get_description_embedding_max(description, emmbed_dict, embed_dim):
    # compute a dimensional maximum over the embeddings of the words in a list
    n = len(description)
    embedding = np.zeros(embed_dim)
    for i, ele in enumerate(description):
        try:
            embedding = np.maximum(embedding, emmbed_dict[ele])
        except:
            pass
    return embedding
    
def cosine(a,b):
    # compute cosine similarity
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def get_embedding_dict (path):
    # read the embeddings dictionary
    emmbed_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            emmbed_dict[word]=vector
    return emmbed_dict

def get_all_embedding(descriptions_, embed_fun, embed_dict, embed_dim ):
    # given a list, compute the embeddings of each element in that list
    vectors =[]
    for ele in tqdm(descriptions_):
        vectors.append(embed_fun(ele,embed_dict,embed_dim))
    return vectors

def get_row_embedding(row, embed_fun, embed_dict, embed_dim ):
    # given a row of the dataset compute the embeddings
    if (type(row)==str):
        list_tokens = ast.literal_eval(row)
        return embed_fun(list_tokens, embed_dict, embed_dim) 
    else:
        return None

def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)

def give_examples_of_clusters(num, clustering):
    # for each cluster give some random examples
    examples = []
    labels = np.unique(clustering.labels_)
    for i in labels:
        examples.append(np.random.choice(find_indices(clustering.labels_,i), size=num))
    return examples
    
def get_embeddings(df):
    # sum the embeddings from indutry tags and keywords tags
    ind_embeddings = df["eng_industries"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_max, embed_glv_50, 50 )).values
    kw_embeddings = df["eng_keywords"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 )).values
    embeddings = []
    for i in range(len(ind_embeddings)):
        if not (ind_embeddings[i] is None) and not (kw_embeddings[i] is None):
            embeddings.append(list(0.5*ind_embeddings[i] + 0.5*kw_embeddings[i]))
        elif not (ind_embeddings[i] is None):
            embeddings.append(list(ind_embeddings[i]))
        elif not (kw_embeddings[i] is None):
            embeddings.append(list(kw_embeddings[i]))
        else:
            embeddings.append(list(np.zeros(50)))
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--df_name', type=str, default='companies_tokenized.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=-1, metavar='N',
                        help='The end integer')
    args = parser.parse_args()

    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')
    tqdm.pandas()
    
    #path of precomputed word embeddings dictionary
    path_glv_50 = 'glove.6B/glove.6B.50d.txt'
    # get the embeddings from path
    embed_glv_50 = get_embedding_dict(path_glv_50)


    tmp_df = companies_df.copy()
    # compute embeddings from tokenized tags
    tmp_df["ind_embed"] = companies_df["tokenized_indutries"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 ))  
    tmp_df["kw_embed"] = companies_df["tokenized_keywords"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 ))  
    
    # combine embeddings from indutry tags and keyword tags
    companies_df["embeddings"] = get_embeddings(tmp_df)
    
    #export the results
    companies_df.to_csv(f'companies_embeddings.csv')