import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import math
from googletrans import Translator 
import requests
import json
import time
from tqdm.auto import tqdm
import math
import os, os.path

import ast
import re

import matplotlib.pyplot as plt


import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import Marker
import warnings 
from geopy.geocoders import Nominatim



import argparse
import os
import urllib.request
from scipy import spatial
from tqdm.autonotebook import tqdm
    


tqdm.pandas()
def my_geocoder(row):
    try:
        point = geolocator.geocode(row).point
        location = geolocator.reverse(str(point.latitude)+","+str(point.longitude))
        address = location.raw['address']
        
        return pd.Series({'Latitude': point.latitude, 'Longitude': point.longitude, 'Full_address': address})
    except:
        return pd.Series({'Latitude': None, 'Longitude': None, 'Full_address': None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--df_name', type=str, default='companies_full 3.csv', metavar='N',
                        help='Name of the dataFrame')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--ending', type=int, default=100, metavar='N',
                        help='The end integer')
    args = parser.parse_args()

    geolocator = Nominatim(user_agent = 'test')
    df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')
    test_df = df.iloc[args.beginning: args.ending, :].copy()
    test_df[['Latitude', 'Longitude', 'Full_address']] = test_df.progress_apply(lambda x: my_geocoder(x['address']), axis=1)
    
    test_df.to_csv(f'geo_{args.beginning}_{args.ending}.csv')