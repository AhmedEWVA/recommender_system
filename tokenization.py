import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import time
from tqdm.auto import tqdm
import math
import os, os.path

import ast
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
# needed for nltk.pos_tag function nltk.download(’averaged_perceptron_tagger’)
nltk.download('wordnet')


from gensim.models import Word2Vec
import gensim
import gensim.downloader

import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors
import seaborn as sns

from collections import Counter

import warnings 

import urllib.request
import argparse
import os
from numpy import dot
from numpy.linalg import norm

def normalize(tokenized_text):
    # takes tokens as input and removes non alphabetical characters
    
    clean_token=[]
    for token in tokenized_text:
        token = token.lower()
        # remove any value that are not alphabetical
        new_token = re.sub(r'[^a-zA-Z]+', ' ', token) 
        # remove empty value and single character value
        if new_token != "" and len(new_token) >= 2: 
            vowels=len([v for v in new_token if v in "aeiou"])
            if vowels != 0: # remove line that only contains consonants
                clean_token.append(new_token)
    return clean_token

def removing_stopwords(normalized_text, stop_words):
     # Extend stop-words
    stop_words.extend(["could","though","would","also","many",'much', 'inc', 'imprint', 'co'])
    # Remove the stopwords from the list of tokens
    tokens = [x for x in normalized_text if x not in stop_words]
    return tokens

def processing_text(description, stop_words):
    # first tokenize
    tokenized_description = WordPunctTokenizer().tokenize(description)
    # second normalize
    normalized_description = normalize(tokenized_description)
    # remove stopwords
    removed_stop_words_description = removing_stopwords(normalized_description, stop_words)

    # if there is an issue do it all over again
    if (len(removed_stop_words_description)==1):
        description = removed_stop_words_description[0]
        tokenized_description = WordPunctTokenizer().tokenize(description)
        normalized_description = normalize(tokenized_description)
        removed_stop_words_description = removing_stopwords(normalized_description, stop_words)
    return removed_stop_words_description

def lemmatizing(removed_stopwords):
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each word and display the output
    lemmatize_text = []
    for word in removed_stopwords:
        output = [word, lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='a'), lemmatizer.lemmatize(word, pos='v')]
        lemmatize_text.append(output)
    return lemmatize_text

def processing_tags(list_of_tags, stop_words):
    # process a list of tags by tokenizing, normalizing and removing stop words
    processed_tags = []
    for e in tqdm(list_of_tags):
        try:
            p = processing_text(e,stop_words)
            processed_tags.append(p)
        except:
            processed_tags.append([])
        
    return processed_tags


def get_tags_corpus(series):
    # this function takes as input a pandas series (a column from a dataframe)
    # aggregate all tags in the series and return a dictionary where keys are tags and values are the numeber of occurances of each tag
    vocab_tags = series.values
    tags_corpus = []
    for industry_tags_row in tqdm(vocab_tags):
        try:
            tags_corpus += ast.literal_eval(industry_tags_row)
        except:
            continue
    return Counter(tags_corpus)

def tokenize_from_dict(dictionary, ele):
    # use a dictionary where the mapping between tags and tokenized tags already exist 
    try:
        #print(ele)
        return dictionary[ele]
    except:
        print(ele)

def tokenize_list_from_dict(dictionary, list_tags):
    if (type(list_tags)==str):
        list_tags = ast.literal_eval(list_tags)
        tokenized_tags = []
        for e in list_tags:
            tokenized_tags += tokenize_from_dict(dictionary, e)
        return tokenized_tags
    else:
        return list_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translating tags')
    parser.add_argument('--df_name', type=str, default='companies_translated.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=-1, metavar='N',
                        help='The end integer')
    args = parser.parse_args()

    stop_words = stopwords.words('english')
    tqdm.pandas()
    #load the dataset where the translation is already done
    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')
    # select a subset of the data
    tags_df = companies_df.loc[:args.ending,["id", "eng_industries", "eng_keywords"]]

    #get unique tags
    unique_ind_tags = get_tags_corpus(tags_df["eng_industries"])
    unique_kw_tags = get_tags_corpus(tags_df["eng_keywords"])

    # tokenize the tags
    tokenized_ind_tags = processing_tags(list(unique_ind_tags.keys()), stop_words)
    tokenized_kw_tags = processing_tags(list(unique_kw_tags.keys()), stop_words)

    # map tags to tokens
    ind_tags_dict = dict(zip(unique_ind_tags, tokenized_ind_tags))
    kw_tags_dict = dict(zip(unique_kw_tags, tokenized_kw_tags))

    #apply the tokenization on all rows and get new columns
    companies_df["tokenized_indutries"] = tags_df["eng_industries"].progress_apply(lambda row : tokenize_list_from_dict(ind_tags_dict,row))
    companies_df["tokenized_keywords"] = tags_df["eng_keywords"].progress_apply(lambda row : tokenize_list_from_dict(kw_tags_dict, row))

    #export the results to new csv file
    companies_df.to_csv(f'companies_tokenized.csv')