import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import math
from googletrans import Translator 
import requests
import json
from tqdm.auto import tqdm
import math
import os, os.path
import ast
import re

from collections import Counter
from gensim.models import Word2Vec
import gensim
import gensim.downloader
import matplotlib.pyplot as plt
import os
from tqdm.autonotebook import tqdm
import argparse
tqdm.pandas()
translator = Translator()
def translate_to_english(desc):
    try:
        lang = detect(desc)
    except:
        print(desc)
        return ''
    if (lang == 'en'):
        return desc
    else :
        try:
            if (type(lang)==str and lang!='nan'):
                return translator.translate(desc, dest='en').text
        except:
            return ''

def translate_tags(tags_list):
    try:
        translated_tags = []
        for tag in tqdm(tags_list):
            translated_tags.append(translate_to_english(tag))
        return translated_tags
    except:
        return tags_list
    
def get_tags_corpus(series):
    vocab_tags = series.values
    tags_corpus = []
    for industry_tags_row in tqdm(vocab_tags):
        try:
            tags_corpus += ast.literal_eval(industry_tags_row)
        except:
            continue
    return Counter(tags_corpus)

def translate_from_dict(dictionary, ele):
    try:
        #print(ele)
        return dictionary[ele]
    except:
        print(ele)

def translate_list_from_dict(dictionary, list_tags):
    if (type(list_tags)==str):
        list_tags = ast.literal_eval(list_tags)
        translated_tags = [translate_from_dict(dictionary, e) for e in list_tags]
        return translated_tags
    else:
        return list_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translating tags')
    parser.add_argument('--df_name', type=str, default='companies_full.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=-1, metavar='N',
                        help='The end integer')
    args = parser.parse_args()

    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')

    tags_df = companies_df.loc[:args.ending,["id", "industries", "keywords"]]

    unique_ind_tags = get_tags_corpus(tags_df["industries"])
    unique_kw_tags = get_tags_corpus(tags_df["keywords"])

    translated_ind_corpus = translate_tags(list(unique_ind_tags.keys()))
    unique_kw_tags_most_common = dict(unique_kw_tags.most_common(300000))
    translated_kw_corpus = translate_tags(list(unique_kw_tags_most_common.keys()))

    ind_tags_dict = dict(zip(unique_ind_tags, translated_ind_corpus))
    kw_tags_dict = dict(zip(unique_kw_tags_most_common, translated_kw_corpus))

    companies_df["eng_indutries"] = tags_df["industries"].progress_apply(lambda row : translate_list_from_dict(ind_tags_dict,row))
    companies_df["eng_keywords"] = tags_df["keywords"].progress_apply(lambda row : translate_list_from_dict(kw_tags_dict, row))

    companies_df.to_csv(f'companies_translated.csv')
