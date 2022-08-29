import requests
import json
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm
import argparse

def process_tags(tags):
    # process that tags to a list of industry tags and a list of keywords tags
    industries = []
    keywords = []
    for i in range(len(tags)):
        if (tags[i]["class"]=="industry" or tags[i]["class"]=="industry_tag"):
            industries.append(tags[i]["name"])
        elif (tags[i]["class"]=="keyword"):
            keywords.append(tags[i]["name"])
    return industries, keywords

def create_item_dict(ele):
    # create an item dictionary
    item = {}
    item["id"] = ele["id"]
    item["name"] = ele["name"]
    item["description"] = ele["description"]
    item["number_of_employees"] = ele["number_of_employees"]
    item["seo_description"] = ele["seo_description"]
    item["annual_revenue_estimation"] = ele["annual_revenue_estimation"]
    item["revenue_currency"] = ele["revenue_currency"]
    item["address"] = ele["address"]
    item["founding_year"] = ele["founding_year"]
    item["headquarter_location_geo_code_jsond"] = ele["headquarter_location_geo_code_json"]
    item["industries"], item["keywords"] = process_tags(ele["tags"])
    return item

def link_to_json(api_url):
    response = requests.get(api_url)
    js = response.json()
    n = len(js['data'])
    items = []
    for i in range (n):
        item = create_item_dict(js['data'][i])
        items.append(item)
    return items

def converting_to_dataframe(json_path, i):
    with open(json_path) as json_file:
        data_list = json.load(json_file)

    df = pd.DataFrame(data_list, columns =["id", "name", "description", "number_of_employees", "seo_description", "annual_revenue_estimation", "revenue_currency", "address", "founding_year", "languages_spoken_at_company", "headquarter_location_geo_code_json", "geo_lon", "geo_lat", "industries", "keywords"])
    df.to_csv(f'csvs/results_{i}.csv', index=False) 
    return df


def factoring_address(address_dict):
    # formating the address into; 'street','zip', 'city', 'state', 'country', 'other'
    elements = ['street','zip', 'city', 'state', 'country', 'other']
    address = ''
    try: 
        for i in elements:
                if not (address_dict[i] == None):
                        address += address_dict[i]+ ", "
        
        return address[:-2]
    except:
        return None

"""def converting_to_dataframe(json_path):
    with open(json_path) as json_file:
        data_list = json.load(json_file)
    for i in range(len(data_list)):
        data_list[i]['address'] = factoring_address(data_list[i]['address'])

    df = pd.DataFrame(data_list, columns =["id", "name", "description", "number_of_employees", "seo_description", "annual_revenue_estimation", "revenue_currency", "address", "founding_year", "languages_spoken_at_company", "headquarter_location_geo_code_json", "geo_lon", "geo_lat", "industries", "keywords"])
 
    return df"""

def combining_to_one_csv(n, directory):
    # concatenating several csv files with similar columns
    csv_path = "csvs/results_0.csv"
    df_full = converting_to_dataframe(csv_path)
    for i in tqdm(list(range(1,n))):
        csv_path = f"csvs/results_{i}.csv"
        df_tmp = pd.read_csv(csv_path)
        df_full = pd.concat([df_full, df_tmp])
    df_full.to_csv(directory, index=False) 
    return df_full


"""combining_to_one_csv(2846,'csvs/companies_full.csv')
len(os.listdir('jsons'))"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geographic localization')
    parser.add_argument('--url', type=int, default = "https://api-tendex.de/api/v1/get-companies-for-enrichment?limit=1000", metavar='N',
                        help='api url')
    parser.add_argument('--beginning', type=int, default=0, metavar='N',
                        help='The beginning interger')
    parser.add_argument('--num_steps', type=int, default=28000, metavar='N',
                        help='The end integer')

    parser.add_argument('--column', type=str, default="ind_embed", metavar='N',
                        help='Specify the column on which the clustering is applied')
    args = parser.parse_args()

    # call the api to get company data and format it
    for i in tqdm(list(range(args.beginning,args.num_steps+args.beginning))):
            t1 = time.time()
            results = link_to_json(args.url)
            df = pd.DataFrame(results, columns =["id", "name", "description", "number_of_employees", "seo_description", "annual_revenue_estimation", "revenue_currency", "address", "founding_year", "languages_spoken_at_company", "headquarter_location_geo_code_json", "geo_lon", "geo_lat", "industries", "keywords"])
            df.to_csv(f'csvs/results_{i}.csv', index=False) 
            """with open(f'jsons/results_{i}.json', 'w') as outfile:
                    json.dump(results, outfile)"""
            t2 = time.time()
    # create a csv file from this
    combining_to_one_csv(args.num_steps,'companies_full.csv')

    