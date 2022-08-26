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

import argparse
from tqdm.autonotebook import tqdm



def distribution_employees(num_employees):
    if (num_employees<50):
        return 1
    elif (num_employees>250):
        return 3
    else:
        return 2

def distribution_revenues(revenues):
    if (revenues<=1000000):
        return 1
    elif (revenues>1000000 and revenues<=10000000):
        return 2
    elif (revenues>10000000 and revenues<=50000000):
        return 3
    elif (revenues>10000000):
        return 4
    else:
        return 5

def range_to_value(range_):
    if range_=='1-10' or range_=='1-9':
        return '10'
    elif range_=='11-50' or range_=='10-49':
        return '30'
    elif range_=='11-100':
        return '50'
    elif range_=='201-500' or range_=='200+' or range_=='101-500':
        return '350'
    elif range_=='51-200' or range_=='50-199':
        return '150'
    else:
        return range_

def get_size(num_employees, revenue):
    revenue = distribution_revenues(revenue)
    num_employees = distribution_employees(num_employees)
    if (revenue ==1 or num_employees==1):
        return 0
    elif (revenue ==4 or num_employees==3):
        return 2
    else: 
        return 1

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--df_name', type=str, default='companies_embeddings.csv', metavar='N',
                        help='Name of the dataFrame')

    args = parser.parse_args()
    tqdm.pandas()  

    companies_df = pd.read_csv(args.df_name,low_memory=False, lineterminator='\n')
    employees = []
    for i in companies_df["number_of_employees"].values:
        try:
            employees.append(float(range_to_value(i)))
        except:
            employees.append(40)
    revenues = []
    for i in companies_df["annual_revenue_estimation"].values:
        try:
            revenues.append(float(range_to_value(i)))
        except:
            revenues.append(1000000)
    sizes = [get_size(employees[i],revenues[i]) for i in range(len(employees))]
    companies_df["size"] = sizes


    companies_df.to_csv(f'companies_with_size.csv')