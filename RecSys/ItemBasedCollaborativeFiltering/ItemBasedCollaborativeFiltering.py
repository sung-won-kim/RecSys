import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# ========================================
# Input 검색할 영화 Title
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--title', required = False, default= 'The Dark Knight Rises',
                    help='검색할 영화 제목 (default : The Dark Knight Rises')
args = parser.parse_args()

# ========================================
# Load Dataset
# ========================================
# load ratings.csv, pre_tmdb_5000_movies.csv from 'https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv' (pre_tmdb_5000_movies : read ContentBasedFiltering.py)
def data_load(dataset):
   data = pd.read_csv(dataset)
   return data

# ========================================
# Preprocessing dataset
# ========================================
rating_data = data_load('./data/ratings_small.csv')
movie_data = data_load('./data/pre_tmdb_5000_movies.csv')
# 필요 없는 열 전처리  
rating_data.drop('timestamp',axis = 1, inplace = True)
# movieid column을 기준으로 dataset merge
movie_data.rename(columns = {'id': 'movieId'}, inplace = True)
user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')

# ========================================
# Item 기반 필터링 추천 (ItemBasedCollaborativeFiltering)
# ========================================
# user-item table을 rating을 속성으로 생성 (아이템 기반 협업 필터링이므로 row가 item, col가 user)
data = user_movie_rating.pivot_table('rating', index = 'title', columns = 'userId').fillna(0)

# ========================================
# Cosine Similarity
# ========================================
movie_sim = cosine_similarity(data, data) # movie_sim : (num_item x num_item) matrix
movie_sim_df = pd.DataFrame(data = movie_sim, index = data.index, columns = data.index)

def SearchMovie(movie_sim_df, name): 
   result = movie_sim_df[name].sort_values(ascending=False)[1:10]
   print(result)
   return result

SearchMovie(movie_sim_df, args.title)