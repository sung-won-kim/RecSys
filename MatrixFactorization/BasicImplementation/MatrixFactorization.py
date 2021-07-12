import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")

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
# user-item table을 rating을 속성으로 생성 (row가 user, col이 item)
data = user_movie_rating.pivot_table('rating', index = 'userId', columns = 'title').fillna(0)
# row가 item, col이 user로 전치
movie_user_rating = data.values.T

# ========================================
# SVD (Singular Value Decomposition)
# ========================================
# python scikit learn의 TruncateSVD는 시그마 행렬의 대각 원소 가운데 상위 n개만 골라줌 then 기존 행렬의 성질을 100%는 아니지만 거의 근사한 값이 나오게 됨
SVD = TruncatedSVD(n_components=12) # latent = 12
matrix = SVD.fit_transform(movie_user_rating) # latent가 12인 형태로 기존 행렬을 변환 (이 경우 (item_num, latent)의 형태로 변환)
corr = np.corrcoef(matrix) # 피어슨 상관계수로 상관도 분석 (이 경우 (item_num, item_num))

# ========================================
# Result
# ========================================
corr_df = pd.DataFrame(data = corr, index = data.columns, columns = data.columns)

def SearchMovie(corr_df, name): 
   result = corr_df[name].sort_values(ascending=False)[1:10]
   print(result)
   return result

SearchMovie(corr_df,args.title)