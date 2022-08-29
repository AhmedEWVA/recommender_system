# Recommender System


This repo is an implementation of the companies recommender system built to be used by the tendex platform. There are mainly nine python files. The output of each python file is a csv file containing the resulting data sets. The files should be executed sequentially with the output of one being the input of another.


&nbsp;
## Requirements
First install the requirements:
``` 
pip install -r requirements.txt
```

&nbsp;
## Contents
- [Fetching data and creating csv file](#fetching-data-and-creating-csv-file)
- [Translation](#translation)
- [Processing text](#processing-text)
- [Embedding](#embedding)
- [Tf-idf](#tf-idf)
- [Size](#size)
- [Geo-data processing](#geo-data-processing)
- [Clustering](#clustering)
- [Recommender](#recommender)

**Note:** All following files include --begining and --ending parameters so that the execution could be done on a smaller portion of the dataset to decrease the processing time.


&nbsp;
## Fetching data and creating csv file

For this the default url is used in the file. In each step 1000 companies are fetched. The output is a csv containing in each row a company and for each column a feature. The selected features are: "id", "name", "description", "number_of_employees", "seo_description", "annual_revenue_estimation", "revenue_currency", "address", "founding_year", "languages_spoken_at_company", "headquarter_location_geo_code_json", "geo_lon", "geo_lat", "industries", "keywords". Features with many missing values could be dropped.
``` 
python fetching_data.py --num_steps=10 
```

&nbsp;
## Translation

This script does the translations to english for the columns: industries and keywords.
``` 
python translation.py --ending=10000  
```

&nbsp;
## Processing text
This script does the tokenization, lemmatization, removal of non alphanumerical characters and stop words.

``` 
python embeddings.py --df_name="companies_translated.csv"  --ending=10000    
```

&nbsp;
## Embedding
This script computes the average word embeddings for the columns: industries and keywords and sum the results to have one embedding vector for each company.
``` 
python embedding.py --df_name="companies_tokenized.csv"  --ending=10000 
```

&nbsp;
## Tf-idf

This script computes a tf-idf vector per company based on the industry tags.
``` 
tf_idf.py --df_name="companies_tokenized.csv"  --ending=10000 
```

&nbsp;
## Size
This script assign one size value for each company based it's number of employees and it's revenues. 0: small, 1: medium and 2: big
``` 
 python size.py --df_name="companies_embeddings.csv" --ending=10000
```

&nbsp;
## Geo-data processing
This script generates the latitude, longitude and full address for company or nan if the address is not found.
``` 
python geo_processing.py --df_name="companies_full.csv" --ending=10000
```

&nbsp;
## Clustering

This script computes the clustering based on embeddings (tfidf could be also specified). There are two clustering methods to choose from. The output is the cluster number for each company
``` 
python clustering.py --df_name="companies_embeddings.csv" --num_clusters=10 --ending=10000
```

&nbsp;
## Recommender
This script generate for each company a list of similar companies. The recommendation is based on the semantic informations (industries and keywords), size and location.
``` 
python recommendations.py --df_name="companies_with_size.csv"  --ending=10000
```