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

import os
import urllib.request
from scipy import spatial
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
import argparse


def get_tags_corpus(series):
    # Given a pandas series (a column from a dataframe)
    # aggregate all tokens in the series and return a dictionary where keys are tags and values are the number of occurances of each tag
    vocab_tags = series.values
    tags_corpus = []
    for industry_tags_row in tqdm(vocab_tags):
        try:
            tags_corpus += ast.literal_eval(industry_tags_row)
        except:
            continue
    return Counter(tags_corpus)

def get_multiple_appearence_words(dict, num_appearances):
    # use only the words that appear more than num_appearances times
    corpus = []
    for i, (key,value) in enumerate(dict.items()):
        if (value>num_appearances):
            corpus.append(key)
    return corpus

def get_number_of_occ(word, description):
    # get the number of occurances of word in description
    description = ast.literal_eval(description)
    i=0
    for ele in description:
        if (ele == word):
            i+=1
    return i

def get_bag_of_word_matrix(series, corpus):
    # compute the bag of words matrix based on corpus and data of each row
    vocab_tags = series.values
    n = len(vocab_tags)
    m = len(corpus)
    mat = np.zeros((n,m))
    for i in tqdm(range(n)):
        for j in range(m):
            try:
                mat[i,j]=get_number_of_occ(corpus[j], vocab_tags[i])
            except:
                continue
    return mat


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
    
    #get corpus by ignoring tokens that appear rarely
    corpus_ind = get_multiple_appearence_words(get_tags_corpus(companies_df["tokenized_indutries"]),10)
    #corpus_kw = get_multiple_appearence_words(get_tags_corpus(companies_df["tokens_keywords"]),10)

    bow_mat_ind = get_bag_of_word_matrix(companies_df["tokenized_keywords"], corpus_ind)
    #bow_mat_kw = get_bag_of_word_matrix(companies_df["tokens_keywords"], corpus_kw)

    # apply tf-idf transformation
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(bow_mat_ind)
    tf_idf_vector=tfidf_transformer.transform(bow_mat_ind)

    # apply pca for dimensionality reduction in order to be able to meaningful compution on the results
    x_ind= tf_idf_vector.toarray()
    x_ind = StandardScaler().fit_transform(x_ind)
    pca_ind = PCA(n_components=5)
    pca_ind.fit(x_ind)
    vec_ind = pca_ind.transform(x_ind)
    vect_ind = list(vec_ind)
    companies_df["ind_tf_idf"] = [list(i) for i in vect_ind]#companies_df["tokens_industries"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 ))  
    #companies_df["kw_tf_idf"] = companies_df["tokens_keywords"].progress_apply(lambda row :  get_row_embedding(row, get_description_embedding_avg, embed_glv_50, 50 ))  

    # export result to a csv file
    companies_df.to_csv(f'companies_tf_idf.csv')